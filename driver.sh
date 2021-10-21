#!/bin/bash

# Save credentials so don't have to enter twice
read -p "Welcome, please enter credentials: " credentials

# Move updated script onto server, replace old scripts in ./scripts dir
echo "Uploading scripts to server, please enter password"
scp MobileNet.py core.py requirements.txt pretrained_weights/* "$credentials":scripts # also prompts password
echo "Done uploading scripts. Now SSH into server, which requires your password again"

# SSH into GPU server
ssh $credentials # will again prompt password