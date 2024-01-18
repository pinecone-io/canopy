# Deploying Canopy on GCP Cloud Run - Step-by-Step Guide


## Introduction
In this guide, we'll walk through the process of deploying Canopy on Google Cloud Platform (`GCP`). 
The steps include setting up GCP, creating a Docker repository, pulling and tagging the `Canopy` image, and finally deploying it using Google Cloud Run.

## Prerequisites
Before you begin, make sure you have the following installed:

- [Docker](https://docs.docker.com/engine/install/)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

and make sure you have a project set up in `GCP`.

## Step 1: Install and Authenticate Google Cloud SDK
Open your terminal and run the following commands:

```bash
# Authenticate with your Google Cloud account
gcloud auth login

# Set the GCP project
gcloud config set project {project-name}
```

## Step 2: Create a Docker Repository on GCP (Optional)
If you have a docker repository in `GCP` you can skip this step. If not run the following commands to create one:

```bash
# Create a Docker repository on GCP
gcloud artifacts repositories create {repository-name} \
    --repository-format docker \
    --location us-west1 \
    --description "Docker repository for storing images of Canopy."
```

## Step 3: Pull and Tag Canopy Docker Image
We'll start by fetching the official `Canopy` image from GitHub Packages, and then we'll apply a tag to prepare the image
for pushing it to Google Cloud Platform (`GCP`). You can access all the available images [here](https://github.com/pinecone-io/canopy/pkgs/container/canopy).

```bash
# Pull the Canopy Docker image
docker pull ghcr.io/pinecone-io/canopy:{canopy-version}

# Tag the image for GCP repository
docker tag ghcr.io/pinecone-io/canopy:{canopy-version} us-west1-docker.pkg.dev/{project-name}/{repository-name}/canopy:{canopy-version}
```

## Step 4: Configure Docker for GCP
```bash
# Configure Docker to use GCP credentials
gcloud auth configure-docker us-west1-docker.pkg.dev
```

## Step 5: Push Canopy Docker Image to GCP
```bash
# Push the Canopy Docker image to GCP repository
docker push us-west1-docker.pkg.dev/{project-name}/{repository-name}/canopy:{canopy-version}
```

## Step 6: Prepare environment variables
Before running the following command make sure to create a `.env` file and include the environment variables mentioned
in [README.md](https://github.com/pinecone-io/canopy/blob/main/README.md). 
```text
OPENAI_API_KEY={open-api-key}
PINECONE_API_KEY={pinecone-api-key}
INDEX_NAME={index-name}

# Other necessary environment variables if needed
```

## Step 7: Creating an Index (Feel free to skip if you already have an index name)

To create a new index in Pinecone, run the following command:

```bash
docker run --env-file .env ghcr.io/pinecone-io/canopy:{canopy-version} yes | canopy new
```

## Step 8: Upserting documents (Feel free to skip if you already have documents)

To upsert documents into Pinecone run:

```bash
docker run --env-file .env ghcr.io/pinecone-io/canopy:{canopy-version} yes | canopy upsert {parquet-file.parquet}
```

## Step 9: Deploy Canopy on Google Cloud Run

To deploy Canopy on GCP, run the following command:

```bash
# Deploy Canopy on Google Cloud Run
gcloud run deploy canopy \
  --image us-west1-docker.pkg.dev/{project-name}/{repository-name}/canopy:{canopy-version} \
  --platform managed \
  --region us-west1 \
  --min-instances 1 \
  --port 8000 \
  --allow-unauthenticated \
  --set-env-vars $(grep -v '^#' .env | tr '\n' ',' | sed 's/,$//')
```

Congratulations! You have successfully deployed `Canopy` on `Google Cloud Run`. 

You should now see an output similar to this:

    Deploying container to Cloud Run service [canopy] in project [project-name] region [us-west1]
    ✓ Deploying new service... Done.
      ✓ Creating Revision...
      ✓ Routing traffic...
      ✓ Setting IAM Policy...
    Done.
    Service [canopy] revision [canopy-00001-6cf] has been deployed and is serving 100 percent of traffic.
    Service URL: https://canopy-bxkpka-uw.a.run.app

## Step 10 - Testing Access to the Service
From your terminal run the following command using the service url you have received from `GCP`:

```bash
curl {service-url}/v1/health
```

If you see the following output your deployment is completed successfully!

```json
{"pinecone_status":"OK","llm_status":"OK"}
```

If you have an issue accessing the server, make sure you have an index. To create an index, follow `Step 7`.

## Step 11 - Chatting with the Service (Optional)
There are several options to interact with your service.

### Canopy's built-in Chat CLI
If you want to use Canopy's built-in Chat CLI, you can run:
```bash
docker run --env-file .env ghcr.io/pinecone-io/canopy:{canopy-version} canopy chat
```
### An OpenAI compatible Chat UI
You can use any OpenAI compatible chat UI to interact with the newly deployed server. 

Examples:
 - [OpenChat](https://github.com/imoneoi/openchat-ui)
 - [Chainlit](https://docs.chainlit.io/get-started/overview)
 - [Hugging Face Chat UI](https://huggingface.co/spaces/huggingchat/chat-ui/blob/main/README.md)

### Using OpenAI's Python Client
You can also use OpenAI's python client to interact with the server. For more information, see
 [Migrating Existing Application to Canopy](https://github.com/pinecone-io/canopy?tab=readme-ov-file#migrating-an-existing-openai-application-to-canopy)
 






