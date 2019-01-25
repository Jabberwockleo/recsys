#!/bin/bash
# File              : build.sh
# Author            : Wan Li
# Date              : 25.01.2019
# Last Modified Date: 25.01.2019
# Last Modified By  : Wan Li

declare -r PROD_DIR="./recsys/"
rm -rf "${PROD_DIR}"
mkdir -p "${PROD_DIR}"
cp -r "./src/"* "${PROD_DIR}"
