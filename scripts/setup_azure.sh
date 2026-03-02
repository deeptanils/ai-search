#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# setup_azure.sh — Provision all Azure resources for the AI Search
#                  Pipeline in a single script.
#
# Prerequisites:
#   - Azure CLI (az) installed and logged in (`az login`)
#   - Sufficient permissions to create resources in the target
#     subscription (Contributor + Cognitive Services Contributor)
#
# Usage:
#   chmod +x scripts/setup_azure.sh
#   ./scripts/setup_azure.sh
#
# The script is idempotent — re-running it skips resources that
# already exist.
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
# Customise these variables or export them before running the script.

RESOURCE_GROUP="${RESOURCE_GROUP:-ai-search-rg}"
LOCATION="${LOCATION:-eastus}"

# Azure AI Foundry (Cognitive Services multi-service account)
FOUNDRY_ACCOUNT_NAME="${FOUNDRY_ACCOUNT_NAME:-ai-search-foundry}"
FOUNDRY_SKU="${FOUNDRY_SKU:-S0}"

# Model deployments
GPT4O_DEPLOYMENT="${GPT4O_DEPLOYMENT:-gpt-4o}"
GPT4O_MODEL="${GPT4O_MODEL:-gpt-4o}"
GPT4O_VERSION="${GPT4O_VERSION:-2024-11-20}"
GPT4O_CAPACITY="${GPT4O_CAPACITY:-30}"

TEXT_EMBED_DEPLOYMENT="${TEXT_EMBED_DEPLOYMENT:-text-embedding-3-large}"
TEXT_EMBED_MODEL="${TEXT_EMBED_MODEL:-text-embedding-3-large}"
TEXT_EMBED_VERSION="${TEXT_EMBED_VERSION:-1}"
TEXT_EMBED_CAPACITY="${TEXT_EMBED_CAPACITY:-120}"

GPT_IMAGE_DEPLOYMENT="${GPT_IMAGE_DEPLOYMENT:-gpt-image-1.5}"
GPT_IMAGE_MODEL="${GPT_IMAGE_MODEL:-gpt-image-1}"
GPT_IMAGE_VERSION="${GPT_IMAGE_VERSION:-2025-04-15}"
GPT_IMAGE_CAPACITY="${GPT_IMAGE_CAPACITY:-1}"

# Cohere Embed v4 is a serverless (Models-as-a-Service) deployment —
# provisioned via the Azure AI Foundry portal, not the CLI.
COHERE_EMBED_MODEL="embed-v-4-0"

# Azure AI Search
SEARCH_SERVICE_NAME="${SEARCH_SERVICE_NAME:-ai-search-service}"
SEARCH_SKU="${SEARCH_SKU:-basic}"
SEARCH_INDEX_NAME="${SEARCH_INDEX_NAME:-candidate-index}"

# Azure Blob Storage
STORAGE_ACCOUNT_NAME="${STORAGE_ACCOUNT_NAME:-aisearchimages}"
STORAGE_CONTAINER_NAME="${STORAGE_CONTAINER_NAME:-images}"

# ── Colours ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Pre-flight checks ───────────────────────────────────────────────
command -v az >/dev/null 2>&1 || fail "Azure CLI (az) is not installed. https://aka.ms/install-azure-cli"
az account show >/dev/null 2>&1 || fail "Not logged in. Run 'az login' first."

SUBSCRIPTION=$(az account show --query name -o tsv)
USER_ID=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || echo "unknown")
info "Subscription : $SUBSCRIPTION"
info "Signed-in ID : $USER_ID"
echo ""

# ── 1. Resource Group ────────────────────────────────────────────────
info "1/6  Resource Group: $RESOURCE_GROUP"
if az group show --name "$RESOURCE_GROUP" >/dev/null 2>&1; then
    ok "Already exists"
else
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none
    ok "Created in $LOCATION"
fi
echo ""

# ── 2. Azure AI Foundry (Cognitive Services) ─────────────────────────
info "2/6  Azure AI Foundry: $FOUNDRY_ACCOUNT_NAME"
if az cognitiveservices account show --name "$FOUNDRY_ACCOUNT_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    ok "Already exists"
else
    az cognitiveservices account create \
        --name "$FOUNDRY_ACCOUNT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --kind "OpenAI" \
        --sku "$FOUNDRY_SKU" \
        --custom-domain "$FOUNDRY_ACCOUNT_NAME" \
        -o none
    ok "Created ($FOUNDRY_SKU)"
fi

FOUNDRY_ENDPOINT=$(az cognitiveservices account show \
    --name "$FOUNDRY_ACCOUNT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query properties.endpoint -o tsv)
FOUNDRY_KEY=$(az cognitiveservices account keys list \
    --name "$FOUNDRY_ACCOUNT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query key1 -o tsv 2>/dev/null || echo "")

# Derive the models/inference endpoint
FOUNDRY_DOMAIN=$(echo "$FOUNDRY_ENDPOINT" | sed 's|https://\(.*\)\.cognitiveservices\.azure\.com/|\1|')
FOUNDRY_EMBED_ENDPOINT="https://${FOUNDRY_DOMAIN}.services.ai.azure.com/models"

info "  Endpoint       : $FOUNDRY_ENDPOINT"
info "  Embed endpoint : $FOUNDRY_EMBED_ENDPOINT"
echo ""

# ── 2a. Model Deployments ───────────────────────────────────────────
deploy_model() {
    local deployment_name=$1 model_name=$2 model_version=$3 capacity=$4 sku_name=${5:-Standard}
    info "  Deploying: $deployment_name ($model_name @ v$model_version, capacity=$capacity)"
    if az cognitiveservices account deployment show \
        --name "$FOUNDRY_ACCOUNT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --deployment-name "$deployment_name" >/dev/null 2>&1; then
        ok "  Already deployed"
    else
        az cognitiveservices account deployment create \
            --name "$FOUNDRY_ACCOUNT_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --deployment-name "$deployment_name" \
            --model-name "$model_name" \
            --model-version "$model_version" \
            --model-format OpenAI \
            --sku-name "$sku_name" \
            --sku-capacity "$capacity" \
            -o none 2>/dev/null || warn "  Deployment may require manual setup (check portal)"
        ok "  Deployed"
    fi
}

info "2a. Model Deployments"
deploy_model "$GPT4O_DEPLOYMENT"      "$GPT4O_MODEL"      "$GPT4O_VERSION"      "$GPT4O_CAPACITY"      "GlobalStandard"
deploy_model "$TEXT_EMBED_DEPLOYMENT"  "$TEXT_EMBED_MODEL"  "$TEXT_EMBED_VERSION"  "$TEXT_EMBED_CAPACITY"  "Standard"
deploy_model "$GPT_IMAGE_DEPLOYMENT"  "$GPT_IMAGE_MODEL"   "$GPT_IMAGE_VERSION"  "$GPT_IMAGE_CAPACITY"   "Standard"
echo ""
warn "Cohere Embed v4 ($COHERE_EMBED_MODEL) is a serverless deployment."
warn "Deploy it manually via the Azure AI Foundry portal:"
warn "  https://ai.azure.com → your project → Model catalog → Cohere → embed-v-4-0 → Deploy"
echo ""

# ── 3. Azure AI Search ──────────────────────────────────────────────
info "3/6  Azure AI Search: $SEARCH_SERVICE_NAME"
if az search service show --name "$SEARCH_SERVICE_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    ok "Already exists"
else
    az search service create \
        --name "$SEARCH_SERVICE_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku "$SEARCH_SKU" \
        -o none
    ok "Created ($SEARCH_SKU)"
fi

SEARCH_ENDPOINT="https://${SEARCH_SERVICE_NAME}.search.windows.net"
SEARCH_KEY=$(az search admin-key show \
    --service-name "$SEARCH_SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query primaryKey -o tsv 2>/dev/null || echo "")

info "  Endpoint : $SEARCH_ENDPOINT"
echo ""

# ── 4. Azure Blob Storage ───────────────────────────────────────────
info "4/6  Blob Storage: $STORAGE_ACCOUNT_NAME"
if az storage account show --name "$STORAGE_ACCOUNT_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    ok "Already exists"
else
    az storage account create \
        --name "$STORAGE_ACCOUNT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2 \
        -o none
    ok "Created (Standard_LRS)"
fi

info "  Creating container '$STORAGE_CONTAINER_NAME'..."
az storage container create \
    --name "$STORAGE_CONTAINER_NAME" \
    --account-name "$STORAGE_ACCOUNT_NAME" \
    --auth-mode login \
    -o none 2>/dev/null && ok "  Container ready" || ok "  Container already exists"
echo ""

# ── 5. RBAC Role Assignments ────────────────────────────────────────
info "5/6  RBAC Role Assignments"
if [[ "$USER_ID" != "unknown" ]]; then
    assign_role() {
        local role=$1 scope=$2
        if az role assignment list --assignee "$USER_ID" --role "$role" --scope "$scope" --query "[0]" -o tsv >/dev/null 2>&1; then
            ok "  $role — already assigned"
        else
            az role assignment create --assignee "$USER_ID" --role "$role" --scope "$scope" -o none 2>/dev/null \
                && ok "  $role — assigned" \
                || warn "  $role — could not assign (check permissions)"
        fi
    }

    FOUNDRY_ID=$(az cognitiveservices account show --name "$FOUNDRY_ACCOUNT_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)
    STORAGE_ID=$(az storage account show --name "$STORAGE_ACCOUNT_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)

    assign_role "Cognitive Services OpenAI User" "$FOUNDRY_ID"
    assign_role "Cognitive Services User"        "$FOUNDRY_ID"
    assign_role "Storage Blob Data Contributor"  "$STORAGE_ID"
else
    warn "  Could not determine signed-in user — assign RBAC roles manually"
fi
echo ""

# ── 6. Generate .env File ───────────────────────────────────────────
info "6/6  Generating .env file"
ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    warn "  $ENV_FILE already exists — writing to .env.generated instead"
    ENV_FILE=".env.generated"
fi

cat > "$ENV_FILE" <<EOF
# Azure AI Foundry
AZURE_FOUNDRY_ENDPOINT=${FOUNDRY_ENDPOINT}
AZURE_FOUNDRY_EMBED_ENDPOINT=${FOUNDRY_EMBED_ENDPOINT}
AZURE_FOUNDRY_API_KEY=${FOUNDRY_KEY}
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=${SEARCH_ENDPOINT}
AZURE_AI_SEARCH_API_KEY=${SEARCH_KEY}
AZURE_AI_SEARCH_INDEX_NAME=${SEARCH_INDEX_NAME}

# Azure Blob Storage
AZURE_STORAGE_ACCOUNT_NAME=${STORAGE_ACCOUNT_NAME}
AZURE_STORAGE_CONTAINER_NAME=${STORAGE_CONTAINER_NAME}

# Azure Computer Vision (optional — Florence backend)
AZURE_CV_ENDPOINT=
AZURE_CV_API_KEY=
EOF

ok "  Written to $ENV_FILE"
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Azure resources provisioned successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Resource Group       : $RESOURCE_GROUP"
echo "  AI Foundry           : $FOUNDRY_ACCOUNT_NAME"
echo "    Endpoint           : $FOUNDRY_ENDPOINT"
echo "    Embed Endpoint     : $FOUNDRY_EMBED_ENDPOINT"
echo "    GPT-4o             : $GPT4O_DEPLOYMENT"
echo "    text-embedding     : $TEXT_EMBED_DEPLOYMENT"
echo "    gpt-image          : $GPT_IMAGE_DEPLOYMENT"
echo "    Cohere embed-v-4-0 : ⚠  Deploy manually via AI Foundry portal"
echo "  AI Search            : $SEARCH_SERVICE_NAME ($SEARCH_SKU)"
echo "    Endpoint           : $SEARCH_ENDPOINT"
echo "    Index              : $SEARCH_INDEX_NAME"
echo "  Blob Storage         : $STORAGE_ACCOUNT_NAME"
echo "    Container          : $STORAGE_CONTAINER_NAME"
echo ""
echo "  .env file            : $ENV_FILE"
echo ""
echo "Next steps:"
echo "  1. Deploy Cohere embed-v-4-0 via the AI Foundry portal"
echo "  2. Review $ENV_FILE and adjust if needed"
echo "  3. source .venv/bin/activate"
echo "  4. python -m ai_search.indexing.cli create"
echo "  5. python scripts/ingest_samples.py --force"
echo ""
