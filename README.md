# DeployIt

DeployIt is a project management and deployment tool that integrates with GitHub, OpenAI, and Milvus to provide advanced features such as code analysis, vector storage, and retrieval-augmented generation (RAG). It helps developers manage projects, analyze code changes, and generate insights using AI-powered tools.

## Features

- **GitHub Integration**: Fetch repository files, analyze git diffs, and manage code changes.
- **OpenAI Integration**: Generate embeddings using OpenAI's `text-embedding-ada-002` model and analyze code changes with GPT-4.
- **Milvus Integration**: Store and query vector embeddings for efficient data retrieval.
- **Project Management**: Create and manage projects with associated GitHub repositories and AI-powered insights.
- **RAG (Retrieval-Augmented Generation)**: Query the vector database to retrieve relevant information for code analysis and insights.

## Setup

### Prerequisites

- Node.js (v16 or higher)
- npm
- A PostgreSQL database
- OpenAI API key
- Milvus vector database

### Set up Git Providers:

To integrate with GitHub and GitLab, you need to create OAuth applications for each provider. Follow the tutorials below to create the apps and configure the required settings:

- **GitHub**:

  - [GitHub OAuth App Tutorial](https://docs.github.com/en/developers/apps/building-oauth-apps/creating-an-oauth-app)
  - **Callback URL**: `${BASE_URL}/api/providers/auth/callback/github`
  - **Scopes**: `read:user`, `user:email`, `repo`, `admin:repo_hook`, `write:repo_hook`, `read:repo_hook`

- **GitLab**:
  - [GitLab OAuth App Tutorial](https://docs.gitlab.com/ee/integration/oauth_provider.html#adding-an-application-through-the-profile)
  - **Callback URL**: `${BASE_URL}/api/providers/auth/callback/gitlab`
  - **Scopes**: `api`, `read_api`, `read_user`, `read_repository`, `write_repository`

### Set up Notion Integration:

To integrate with Notion, you need to create a Notion integration and connect the required databases to it. Follow the tutorials below to configure the required settings:

- **Create a Notion Integration**:

  - [Notion Integration Tutorial](https://developers.notion.com/docs/create-a-notion-integration)
  - **Integration Token**: After creating the integration, copy the integration token and add it to your `.env` file as `NOTION_CLIENT_ID` and `NOTION_CLIENT_SECRET`.

- **Connect Databases to the Integration**:
  - [Connect a Database to a Notion Integration](https://developers.notion.com/docs/working-with-databases#sharing-a-database-with-your-integration)
  - **Note**: Ensure that the databases you want DeployIt to access are connected with the integration. Without this step, DeployIt will not be able to access the databases.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/deployit.git
cd deployit
```

2. Install dependencies:

```bash
npm install
```

3. Set up environment variables:

Create a `.env` file in the root directory and add the following variables:

```env
DATABASE_URL=your_postgresql_database_url
GITHUB_CLIENT_ID=github_oath_app_client_id
GITHUB_CLIENT_SECRET=github_oath_app_client_secret
GITLAB_CLIENT_ID=gitlab_oath_app_client_id
GITLAB_CLIENT_SECRET=gitlab_oath_app_client_secret
OPENAI_API_KEY=your_openai_api_key
MILVUS_ADDRESS=your_milvus_address
MILVUS_USERNAME=your_milvus_username
MILVUS_PASSWORD=your_milvus_password
NOTION_CLIENT_ID=notion_integrations_client_id
NOTION_CLIENT_SECRET=notion_integrations_client_secret
BASE_URL= app_base_url
DEV_WEBHOOK_CALLBACK= webhook_url // for dev only
SMTP_HOST=your_smtp_host
SMTP_PORT=your_smtp_port
SMTP_USER=your_smtp_user_email
SMTP_PASS=your_smtp_user_password
```
For `DEV_WEBHOOK_CALLBACK`, Webhooks require a publicly accessible URL for the webhook provider to send event payloads. You can use Hookdeck to provide a public URL for the callback.


4. Run Prisma migrations:

```bash
npx prisma migrate dev
npx prisma db pull
npx prisma generate
```

## Development

Start the development server on `http://localhost:3000`:

```bash
npm run dev
```

## Production

Build the application for production:

```bash
npm run build
```

Locally preview the production build:

```bash
npm run preview
```

Check out the [Nuxt deployment documentation](https://nuxt.com/docs/getting-started/deployment) for more information on deploying the app.

## Technologies Used

- **Nuxt.js**: Frontend framework for building the user interface.
- **Prisma**: ORM for managing the PostgreSQL database.
- **OpenAI**: AI-powered embeddings and code analysis.
- **Milvus**: Vector database for storing and querying embeddings.
- **GitHub API**: Integration for fetching repository files and analyzing code changes.

## Acknowledgments

- [Nuxt.js Documentation](https://nuxt.com/docs/getting-started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Milvus Documentation](https://milvus.io/docs/)
- [Prisma Documentation](https://www.prisma.io/docs/)
