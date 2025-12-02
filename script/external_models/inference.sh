# NOTE: This file has been ARCHIVED. Do not use this file.

#!/bin/bash
# source env/bin/activate

helpFunction()
{
   echo ""
   echo "Usage: $0 -m model -d dataset"
   echo -e "\t-m Inference model to use"
   echo -e "\t-d Dataset to perform inference on"
   exit 1 # Exit script after printing help
}

while getopts "m:d:" opt
do
    case "$opt" in
        m) model="$OPTARG" 
        ;;
        d) dataset="$OPTARG" 
        ;;
        ?) echo "-$OPTARG is not a valid option: ignoring..."; # Ignore any invalid options
        ;;
        :) echo "Option -$OPTARG is missing an argument"; # Send the help function if a parameter is missing an argument
            helpFunction
        ;;
    esac
done

# Check which model we are using
if [[ -z "$model" ]]; then
    echo "An inference model has not been provided.";
    helpFunction
elif [[ "$model" == "depth-anything" || "$model" == "da" ]]; then
    model="depth-anything"
elif [[ "$model" == "pixel perfect" || "$model" == "pdd" ]]; then
    dataset="pixel perfect"
elif [[ "$model" == "sharpdepth" || "$model" == "sd" ]]; then
    model="sharpdepth"
else
    echo "Invalid model provided."
    echo "Here are the valid models:"
    echo -e "\t- depth-anything [da]"
    echo -e "\t- pixel perfect [pdd]"
    echo -e "\t- sharpdepth [sd]"
    exit 1
fi

# Check which dataset we are using
if [[ -z "$dataset" ]]; then
    dataset="unspecified"
elif [[ "$dataset" == "hypersim" || "$dataset" == "hyper" || "$dataset" == "h" ]]; then
    dataset="hypersim"
elif [[ "$dataset" == "nyu" || "$dataset" == "n" ]]; then
    dataset="nyu"
elif [[ "$dataset" == "middlebury" || "$dataset" == "middle" || "$dataset" == "m" ]]; then
    dataset="middlebury"
elif [[ "$dataset" == "kitti" || ]]
else
    echo "An invalid dataset was provided: running as if dataset was not specified..."
    dataset="unspecified"
fi

# Begin script in case all parameters are correct
if [[ "$dataset" != "unspecified" ]]; then
    echo "Running $model on $dataset..."
else
    echo "Running $model on everything..."
fi

cd ../..

if [[ "$model" == "depth-anything" ]]; then
    ./script/external_models/run-depth-anything.sh -d "$dataset" -t
elif [[ "$model" == "pixel perfect" ]]; then
    ./script/external_models//run-ppd.sh -d "$dataset" -t
elif [[ "$model" == "sharpdepth" ]]; then
    ./script/external_models//run-sharpdepth.sh -d "$dataset" -t
fi