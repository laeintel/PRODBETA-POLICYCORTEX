#!/bin/bash

# Update application-deploy.yml
sed -i 's/crpolicycortexdev/crpolicortex001dev/g' .github/workflows/application-deploy.yml
sed -i 's/crpolicycortex/crpolicortex001/g' .github/workflows/application-deploy.yml
sed -i 's/rg-policycortex-/rg-policortex001-/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-api_gateway/policortex001-api_gateway/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-azure_integration/policortex001-azure_integration/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-ai_engine/policortex001-ai_engine/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-data_processing/policortex001-data_processing/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-conversation/policortex001-conversation/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-notification/policortex001-notification/g' .github/workflows/application-deploy.yml
sed -i 's/policycortex-frontend/policortex001-frontend/g' .github/workflows/application-deploy.yml
sed -i "s/tags.project=='policycortex'/tags.project=='policortex'/g" .github/workflows/application-deploy.yml

# Update bicep-deploy.yml
sed -i 's/rg-policycortex-/rg-policortex001-/g' .github/workflows/bicep-deploy.yml
sed -i 's/crpolicycortex/crpolicortex001/g' .github/workflows/bicep-deploy.yml

echo "YAML files updated with 001 suffix"