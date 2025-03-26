import OpenAI from "openai";
import { querySourceType, getAllEntries, insertSource } from '~~/server/utils/milvus';
import { encoding_for_model } from 'tiktoken';
import { max } from "lodash";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY ?? 'apiKey',
});

const tokenLimit = 8000;
const encoding = encoding_for_model("gpt-4");
const encodingModel = 'text-embedding-ada-002';

function countTokens(text: string): number {
    if (typeof text !== 'string') {
        console.error('Invalid input to countTokens. Expected a string but received:', typeof text);
        throw new Error('Invalid input to countTokens: Input must be a string.');
    }

    const tokenizer = encoding_for_model(encodingModel);
    return tokenizer.encode(text).length;
}

/**
 * Counts the tokens of a vector or an array of vectors using tiktoken.
 * @param vectors - A single vector (`number[]`) or multiple vectors (`number[][]`).
 * @returns The total token count.
 */
function countVectorTokens(vectors: number[] | number[][]): number {
    const tokenizer = encoding_for_model(encodingModel);

    if (Array.isArray(vectors[0])) {
        // If it's a 2D array (number[][]), flatten it into a single array
        const flattenedVector = (vectors as number[][]).flat();
        const vectorString = flattenedVector.join(' '); // Convert to a string
        return tokenizer.encode(vectorString).length;
    } else {
        // If it's a 1D array (number[]), convert it to a string
        const vectorString = (vectors as number[]).join(' ');
        return tokenizer.encode(vectorString).length;
    }
}

export async function createEmbedding(content: string) {
    const embedding = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: content,
    });

    if (!embedding || !embedding.data || !embedding.data[0] || !Array.isArray(embedding.data[0].embedding)) {
        throw new Error('Failed to generate a valid embedding from OpenAI.');
    }

    return embedding.data[0].embedding;
}

function splitIntoChunks(text: string, maxTokens: number=tokenLimit/2): string[] {
    const words = text.split(/\s+/); // Split by whitespace
    const chunks = [];
    for (let i = 0; i < words.length; i += maxTokens) {
        const chunk = words.slice(i, i + maxTokens).join(' ');
        if (chunk.trim() !== '') { // Ensure the chunk is not empty
            chunks.push(chunk);
        }
    }
    return chunks;
}

export async function createChunkEmbedding(content: string): Promise<number[][]> {
    // Split content into chunks
    const chunks = splitIntoChunks(content);
    const embeddings: number[][] = [];

    for (const chunk of chunks) {
        try {
            const embedding = await createEmbedding(chunk);
            embeddings.push(embedding);
        } catch (error) {
            console.error('Error generating embedding for chunk:', chunk, error);
            throw new Error('Failed to generate embedding for one of the chunks.');
        }
    }

    // Validate the final embeddings
    if (!Array.isArray(embeddings) || embeddings.some((vector) => vector.some((value) => typeof value !== 'number' || isNaN(value)))) {
        throw new Error('Invalid content: content must be a 2D array of numbers.');
    }

    return embeddings;
}

async function getLLMResponse(messages: any[], maxTokens: number=tokenLimit): Promise<string | undefined> {
    try {
        const completion = await openai.chat.completions.create({
            model: "gpt-4",
            messages,
            max_tokens: maxTokens,
        });

        return completion.choices[0]?.message?.content;
    } catch (error: any) {
        console.error('Error in getLLMResponse:', error.response?.data || error.message || error);
        throw new Error('Failed to get a response from OpenAI.');
    }
}

/**
 * Summarizes the given text if it exceeds the specified token limit.
 */
export async function summarizeContent(
    content: string,
    topK: number = 5,
): Promise<string> {

    const maxTokens = tokenLimit / 2;

    // Tokenize the content using tiktoken
    const tokens = encoding.encode(content);

    // If the token count exceeds the limit, truncate it
    if (tokens.length > maxTokens) {
        console.log(`Content exceeds ${maxTokens} tokens, truncating...`);
        // Truncate to the maximum allowed number of tokens
        tokens.splice(maxTokens);
    }

    // Convert tokens back to text
    const truncatedContent = encoding.decode(tokens);

    // Now, let's ensure that the truncated content fits within the maxTokens
    let resultLimit = maxTokens - tokens.length;

    // Prepare the messages for the summarization request
    const messages = [
        {
            role: "system",
            content: `You are a summarization assistant. Please summarize the following content so that it fits within ${maxTokens} tokens, while preserving its key information.`
        },
        {
            role: "user",
            content: truncatedContent
        }
    ];

    // Make the request for summarization
    const summary = await getLLMResponse(messages, resultLimit);

    if (summary) {
        const summaryTokens = encoding.encode(summary).length;
        console.log('Final token count for summary:', summaryTokens);

        // Truncate the summary if it exceeds the resultLimit
        if (summaryTokens > resultLimit) {
            console.warn(`Generated summary exceeds the resultLimit of ${resultLimit} tokens. Truncating...`);
            return summary.slice(0, Math.floor(resultLimit * 4)).trim(); // Approximation: 4 characters per token
        }

        return summary.trim();
    }

    throw new Error('Failed to generate project summary.');
}


/**
 * Determines if a file is logically valuable by excluding files such as packages or vendor modules.
 */
function isValidCodeFile(fileName: string): boolean {
    const lowerFileName = fileName.toLowerCase();
    const excludePatterns = ["node_modules", "package.json", "package-lock.json", "yarn.lock", "tsconfig", ".min.js", "vendor"];
    return !excludePatterns.some(pattern => lowerFileName.includes(pattern));
}

/**
 * Generates a natural language explanation for a single source file.
 * @param fileContent - The content of the source file.
 * @param fileName - The name of the file.
 */
async function generateNLExplanationForFile(fileContent: string, fileName: string): Promise<string> {
    let contentToProcess = String(fileContent);
    contentToProcess = contentToProcess.replace(/\n+/g, ' ');

    // Split content into chunks if it exceeds the token limit
    const maxTokens = tokenLimit / 2; // Set a reasonable limit for each chunk
    const tokens = countTokens(contentToProcess);

    let chunkSummaries: string[] = [];

    if (tokens > maxTokens) {
        console.log(`File "${fileName}" exceeds ${maxTokens} tokens. Splitting into chunks...`);
        const chunks = splitIntoChunks(contentToProcess, maxTokens);
        const chunkLimit = maxTokens / chunks.length
        // Summarize each chunk
        for (const chunk of chunks) {
            const chunkMessages = [
                {
                    role: "system",
                    content: `You are a software analysis assistant. Analyze the following chunk of source code from the file "${fileName}" and generate a clear, plain language explanation of its main functionalities and purpose.`
                },
                {
                    role: "user",
                    content: chunk
                }
            ];

            const chunkSummary = await getLLMResponse(chunkMessages, chunkLimit);
            console.log('asdfasdfsadfasdfasdfsdfsdf')
            if (chunkSummary) {
                chunkSummaries.push(chunkSummary.trim());
            }
        }

        // Combine all chunk summaries into a single explanation
        contentToProcess = chunkSummaries.join('\n\n');
    }
    
    // Generate a final explanation for the combined summaries or the original content
    const finalMessages = [
        {
            role: "system",
            content: `You are a software analysis assistant. Analyze the following source code from the file "${fileName}" and generate a clear, plain language explanation of its main functionalities and overall purpose. Focus on the logical value of the code.`
        },
        {
            role: "user",
            content: contentToProcess
        }
    ];

    const explanation = await getLLMResponse(finalMessages, maxTokens);
    return explanation ? explanation.trim() : "";
}

/**
 * Processes the initial import of the codebase:
 * - Iterates over all source code files in the RAG,
 * - Filters out files that are not valuable,
 * - Generates natural language explanations for each valid file,
 * - And stores these explanations in the RAG.
 */
export async function processInitialImportForNLExplanations(collectionName: string, repositoryId: number, files: any[]): Promise<void> {    
    for (const entry of  files) {
        const fileContent = entry.content;
        const fileName = entry.fileName || "Unnamed File";
        
        if (!isValidCodeFile(fileName)) {
            continue;
        }
        
        const nlExplanation = await generateNLExplanationForFile(fileContent, fileName);
        const vector = await createEmbedding(nlExplanation)
        await insertSource(collectionName, repositoryId, 'nl_explanation', fileName, [vector], nlExplanation);
    }
}
  
/**
 * Generates global summaries from the NL explanations stored in the RAG.
 * This includes:
 *  - A global overview of the software,
 *  - A detailed list of features.
 */
export async function generateGlobalSummaries(collectionName: string): Promise<{ globalOverview: string, detailedFeatures: string }> {
    // Retrieve all NL explanation entries.
    const explanationEntries = await querySourceType(collectionName, 'nl_explanation');
    let combinedExplanations = explanationEntries.data.map((entry: any) => entry.content).join('\n\n');
    
    if (countTokens(combinedExplanations) > tokenLimit) {
      combinedExplanations = await summarizeContent(combinedExplanations);
    }
    
    const globalMessages = [
      {
        role: "system",
        content: `You are a software documentation assistant. Based on the provided natural language explanations of individual source code files, generate a comprehensive overview of the software. Explain its overall purpose, main functionalities, and architecture in plain language.`
      },
      {
        role: "user",
        content: combinedExplanations
      }
    ];
    
    const globalOverview = await getLLMResponse(globalMessages) || "";
    
    const featureMessages = [
      {
        role: "system",
        content: `You are a software analysis assistant. From the provided natural language explanations, extract and list all the detailed features of the software. For each feature, provide a brief functional description in plain language.`
      },
      {
        role: "user",
        content: combinedExplanations
      }
    ];
    
    const detailedFeatures = await getLLMResponse(featureMessages) || "";
    
    return {
      globalOverview: globalOverview.trim(),
      detailedFeatures: detailedFeatures.trim()
    };
}
  
/**
 * Updates the natural language documentation using a provided diff.
 * Process:
 *  - Extract a broader code context from the diff,
 *  - Identify which functionalities are affected,
 *  - Retrieve the NL explanation documents corresponding to these features,
 *  - Generate updated explanations reflecting the changes,
 *  - And store the updated documentation in the RAG.
 */
export async function updateNLDocumentationWithDiff(collectionName: string, diffContent: string): Promise<void> {
    // Step 1: Extract a broader code context from the diff.
    const contextMessages = [
      {
        role: "system",
        content: `You are a code context extraction assistant. Based on the following diff, extract the larger code context surrounding the changes.`
      },
      {
        role: "user",
        content: diffContent
      }
    ];
    const codeContext = await getLLMResponse(contextMessages) || "";
    
    // Step 2: Identify the affected features based on the extracted context.
    const featureMessages = [
      {
        role: "system",
        content: `You are a software analysis assistant. Analyze the following code context extracted from a diff and identify which functionalities of the software are affected. Provide a list of feature names or descriptions in plain language.`
      },
      {
        role: "user",
        content: codeContext
      }
    ];
    const affectedFeatures = await getLLMResponse(featureMessages) || "";
    
    // Step 3: Retrieve the NL explanation documents corresponding to the affected features.
    const keywords = affectedFeatures.split(',').map((kw: string) => kw.trim());
    let relevantExplanations: string[] = [];
    const explanationEntries = await querySourceType(collectionName, 'nl_explanation');
    for (const entry of explanationEntries.data) {
      for (const kw of keywords) {
        if (entry.content.toLowerCase().includes(kw.toLowerCase())) {
          relevantExplanations.push(entry.content);
          break;
        }
      }
    }
    
    let combinedRelevantExplanations = relevantExplanations.join('\n\n');
    if (countTokens(combinedRelevantExplanations) > tokenLimit) {
      combinedRelevantExplanations = await summarizeContent(combinedRelevantExplanations);
    }
    
    // Step 4: Generate updated documentation for the affected features using the diff.
    const updateMessages = [
      {
        role: "system",
        content: `You are a software documentation assistant. Given the current documentation for certain software features and the changes indicated by a diff, update the natural language explanations of these features to reflect the new state of the code. Provide the updated explanations in plain language.`
      },
      {
        role: "user",
        content: `Current feature documentation:\n${combinedRelevantExplanations}\n\nDiff changes:\n${diffContent}`
      }
    ];
    const updatedDocumentation = await getLLMResponse(updateMessages) || "";
    
    // Step 5: Store the updated documentation in the RAG.
    const currentTime = Date.now();
    const updateEntry = {
      id: currentTime,
      repository_id: 0,
      source: "Updated Features Documentation",
      type: 'nl_explanation',
      content: updatedDocumentation.trim(),
    };
    
    await insertSource(collectionName, updateEntry.repository_id, updateEntry.type, updateEntry.source, null, [[updateEntry.content]], null);
}

export async function diffPrompt(
    collectionName: string,
    diff: any,
    topK: number = 10
  ) {

    // Create a query vector using a representative technical phrase
    const queryText = "Software engineer analysis of the diff";
    const queryVector = await createEmbedding(queryText);
    
    // Retrieve only the topK most relevant RAG sources for source code analysis
    const ragSources = await queryVectors(collectionName, queryVector, topK);
    let ragSourceContent = ragSources.results.map((source: any) => source.con).join('\n');
    
    // Check token count and summarize if needed
    if (countTokens(ragSourceContent) > tokenLimit) {
      ragSourceContent = await summarizeContent(ragSourceContent);
    }

    diff = JSON.stringify(diff);

    const messages = [
        {
            role: "system",
            content: `You are an assistant specializing in software engineering tasks. You have just received a git diff of the codebase. The diff contains the changes made to the software. Your task is to review the diff and check if it represents a major advancement of the software. If so, identify the main changes. Return the response in the following JSON format:
            {
                "isMajorAdvancement": boolean,
                "mainChanges": [string]
            }
            Here's the codebase in vector: ${ragSourceContent} and the changes are: ${diff}`
        },
        {
            role: "user",
            content: "Analyze the diff and return the response in JSON format."
        },
        {
            role: "user",
            content: `List 3 SEO articles with 100 words that could help future users in the reflection phase to understand these developments. Add the list in a new property 'seoArticleSuggestions' with the following JSON format:
                "seoArticleSuggestions": [{
                    "title": string,
                    "content": string
                }]
            `
        },
    ]

    const responseContent = await getLLMResponse(messages)
    await updateNLDocumentationWithDiff(collectionName, diff)
    
    if (responseContent) {
        const cleanedResponse = responseContent.trim();
        try {
          return JSON.parse(cleanedResponse);
        } catch (parseError) {
          console.error('Error parsing response:', parseError);
          throw new Error('Failed to parse response from OpenAI.');
        }
    }
      
    return { status: 'error', message: 'No response from OpenAI.' };

}

/**
 * Generate a detailed explanation of the software based on its complete source code.
 * 
 * This function performs the following steps:
 * 1. Retrieve all code source entries from the RAG.
 * 2. For each file (or code entry), generate a plain language summary of its main functionalities.
 * 3. Combine the file summaries and, if necessary, recursively summarize them.
 * 4. Produce a final detailed explanation of the overall software.
 */
export async function generateSoftwareExplanation(
    collectionName: string
  ): Promise<string> {
    // Step 1: Retrieve all code entries from the RAG for source code
    const codeEntries = await querySourceType(collectionName, 'source_code');
    const fileSummaries: string[] = [];
  
    // Step 2: Process each file/entry individually
    for (const entry of codeEntries.data) {
      const fileContent = entry.content;
      // Use fileName if provided, otherwise a generic identifier
      const fileName = entry.fileName || "Unnamed File";
      
      // Ensure content does not exceed token limit; summarize if necessary
      let contentToProcess = fileContent;
      if (countTokens(contentToProcess) > tokenLimit) {
        contentToProcess = await summarizeContent(contentToProcess);
      }
      
      // Create a prompt to extract functionalities for this file
      const messages = [
        {
          role: "system",
          content: `You are a software analysis assistant. Analyze the following source code from the file "${fileName}" and provide a plain language summary of its main functionalities. Describe its role and contribution to the overall software in clear, everyday language.`
        },
        {
          role: "user",
          content: contentToProcess
        }
      ];
      
      const fileSummary = await getLLMResponse(messages);
      if (fileSummary) {
        fileSummaries.push(`File: ${fileName}\nSummary: ${fileSummary.trim()}`);
      }
    }
  
    // Step 3: Combine all file summaries into one text block
    let combinedSummaries = fileSummaries.join('\n\n');
  
    // If the combined text exceeds the token limit, recursively summarize it
    if (countTokens(combinedSummaries) > tokenLimit) {
      combinedSummaries = await summarizeContent(combinedSummaries);
    }
  
    // Step 4: Generate the final explanation based on the combined summaries
    const finalMessages = [
      {
        role: "system",
        content: `You are a software documentation assistant. Given the combined summaries of individual source code files, generate a detailed explanation of how the software works. Your explanation should be in plain language and cover the essential functionalities and overall architecture of the software.`
      },
      {
        role: "user",
        content: combinedSummaries
      }
    ];
  
    const finalExplanation = await getLLMResponse(finalMessages);
    return finalExplanation ? finalExplanation.trim() : "Failed to generate software explanation.";
}

function chunkContent(content: string, maxChars: number): string[] {
    const chunks: string[] = [];
    let currentChunk = '';

    for (const line of content.split(' ')) {
        // Check if adding the current line exceeds the maxChars
        // If the current line exceeds maxChars, push the current chunk and start a new one
        if (line.length > maxChars) {
            if (currentChunk.trim().length > 0) {
                chunks.push(currentChunk.trim());
            }
            // If a single line exceeds the limit, split it into smaller chunks
            let startIdx = 0;
            while (startIdx < line.length) {
                const endIdx = Math.min(startIdx + maxChars, line.length);
                chunks.push(line.slice(startIdx, endIdx));
                startIdx = endIdx;
            }
            currentChunk = ''; // Reset currentChunk
        } else if ((currentChunk + line).length > maxChars) {
            // If adding the line exceeds maxChars, push the current chunk and start a new one
            chunks.push(currentChunk.trim());
            currentChunk = line;
        } else {
            // Otherwise, add the line to the current chunk
            currentChunk += line + ' ';
        }
    }

    // Push the last chunk if it has content
    if (currentChunk.trim() !== '') {
        chunks.push(currentChunk.trim());
    }

    return chunks;
}


async function summarizeChunk(text: string, remainingTokens: number): Promise<string> {

    const messages = [
        {
            role: "system",
            content: `You are a software analysis assistant. Analyze the following content and generate a concise summary of its key features, functionalities, and purpose. Focus on the logical structure and main points.`
        },
        {
            role: "user",
            content: text //NL files
        }
    ];

    const summary = await getLLMResponse(messages, remainingTokens);  // Use the appropriate method to call the model
    return summary?.trim() || "";
}

export async function extractProjectSummary(
    collectionName: string,
    topK: number = 5,
): Promise<string> {

    const maxTokens = tokenLimit / 2;  // To avoid large chunks, set a reasonable token limit per chunk

    const ragSources = await querySourceType(collectionName, 'nl_explanation');

    const truncatedSources: string[] = [];

    for (const source of ragSources.data) {
        let sourceContent = source.summary;

        // Ensure source content is in string form
        if (typeof sourceContent !== "string") {
            sourceContent = String(sourceContent);
        }

        const tokens = encoding.encode(sourceContent);

        if (tokens.length > maxTokens) {
            // If the content exceeds maxTokens, chunk it into smaller parts
            const chunks = chunkContent(sourceContent, maxTokens);
            for (const chunk of chunks) {
                    const summarizedChunk = await summarizeChunk(chunk, maxTokens);
                    truncatedSources.push(summarizedChunk);
            }
        } else {
            const summarizedChunk = await summarizeChunk(sourceContent, maxTokens);
            truncatedSources.push(summarizedChunk);
        }
    }

    // Combine all the truncated chunks into the final content
    const codeContent = truncatedSources.join('\n');

    const resultLimit = tokenLimit - countTokens(codeContent);
    const messages = [
        {
            role: "system",
            content: `You are a software analysis assistant. Analyze the following source code snippets and summarize their key features, functionalities, and purpose. Ensure the summary is concise and fits within ${resultLimit} tokens.`
        },
        {
            role: "user",
            content: codeContent
        }
    ];

    // Generate the final summary based on the combined content
    const summary = await getLLMResponse(messages, resultLimit);

    if (summary) {
        const summaryTokens = encoding.encode(summary).length;

        if (summaryTokens > resultLimit) {
            console.warn(`Generated summary exceeds the resultLimit of ${resultLimit} tokens. Truncating...`);
            return summary.slice(0, Math.floor(resultLimit * 4)).trim(); // Approximation: 4 characters per token
        }

        return summary.trim();
    }

    throw new Error('Failed to generate project summary.');
}

export async function projectInsightsPrompt(
    collectionName: string,
    topK: number = 10
) {
    try {
        // Extract a summary from the source code
        const projectSummary = await extractProjectSummary(collectionName, topK);
    
        const marketingInsights = await marketingInsightsPrompt(projectSummary)
        const technicalInsights = await technicalInsightsPrompt(projectSummary)

        return { marketingInsights, technicalInsights}
    } catch (parseError) {
        console.error('Error parsing response:', parseError);
        throw new Error('Failed to parse response from OpenAI.');
    }
}

export async function marketingInsightsPrompt( summary: string ) {
    const message = `Here is the project summary extracted from the source code: ${summary}`
    const maxLimit = tokenLimit - countTokens(message)
    
    const messages = [
        {
            role: "system",
            content: `You are a marketing assistant specialized in converting technical summaries into marketing insights. Analyze the following project summary and return a JSON object with the structure:
            {
                "projectOverview": string,
                "keyFeatures": [string],
                "mainValueProposition": string,
                "sampleProductIntroductionEmail": {
                    "subject": string,
                    "body": string
                }
            }
            Do not wrap the JSON in any markdown or code blocks.`
        },
        {
            role: "user",
            content: message
        }
    ];
    
    const responseContent = await getLLMResponse(messages,maxLimit);

    if (responseContent) {
      const cleanedResponse = responseContent.trim();
      try {
        return JSON.parse(cleanedResponse);
      } catch (parseError) {
        console.error('Error parsing response:', parseError);
        throw new Error('Failed to parse response from OpenAI.');
      }
    }

    throw new Error('No response from OpenAI.');
}

export async function technicalInsightsPrompt( summary: string) {
    const message = `Here is the project summary extracted from the source code: ${summary}`
    const maxLimit = tokenLimit - countTokens(message)
    
    const messages = [
        {
            role: "system",
            content: `You are a technical assistant specialized in converting technical summaries into technical insights. Analyze the following project summary and return a JSON object with the structure:
            {
                "programmingLanguages": [string],
                "technologies": [string],
                "architecturalOverview": string,
                "miniAudit": {
                    "improvementSuggestions": [string]
                },
                "identifiedLimitations": [string]
            }
            Do not wrap the JSON in any markdown or code blocks.`
        },
        {
            role: "user",
            content: message
        }
    ];
    
    const responseContent = await getLLMResponse(messages,maxLimit);

    if (responseContent) {
      const cleanedResponse = responseContent.trim();
      try {
        return JSON.parse(cleanedResponse);
      } catch (parseError) {
        console.error('Error parsing response:', parseError);
        throw new Error('Failed to parse response from OpenAI.');
      }
    }

    throw new Error('No response from OpenAI.');
}
