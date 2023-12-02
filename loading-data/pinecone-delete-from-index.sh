#!/bin/zsh

PINECONE_API_KEY=$1
PINECONE_HOST=$2

curl --request POST \
     --url $PINECONE_HOST/vectors/delete \
     --header "Api-Key: $PINECONE_API_KEY" \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '
{
  "deleteAll": "true"
}
'