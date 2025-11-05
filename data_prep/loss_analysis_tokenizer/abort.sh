#!/bin/bash

echo "Cancelling loss analysis jobs..."
squeue -u $USER -h -o "%i %j" | awk '/loss-(prep|base|extended)/ {print $1}' | xargs -r scancel
echo "Done."
