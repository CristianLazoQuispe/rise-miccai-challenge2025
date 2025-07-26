import synapseclient 
import synapseutils
from dotenv import load_dotenv
import os
import os
import shutil
from tqdm import tqdm
import os
import shutil
from tqdm import tqdm
# Folder paths
source_dir = '/home/va0831/.synapseCache/'
destination_dir = '/data/cristian/projects/med_data/rise-miccai/'


# Load environment variables from .env file
load_dotenv()
# Authenticate with Synapse using the token from the environment variable
TOKEN_SYNAPSE = os.getenv('TOKEN_SYNAPSE')

if not os.path.exists(destination_dir):
    print("Using Synapse token:", TOKEN_SYNAPSE[:4])
    syn = synapseclient.Synapse() 
    syn.login(authToken=TOKEN_SYNAPSE) 
    files = synapseutils.syncFromSynapse(syn, 'syn68646516') 

    #[syn68647084]: Downloaded to /home/va0831/.synapseCache/818/160229818/LISA_0043_ciso.nii.gz


    # List all files and directories in the source directory
    items = os.listdir(source_dir)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move the files and directories with a progress bar
    for item in tqdm(items, desc="Moving items", unit="item"):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(destination_dir, item)
        
        # Check if it's a file or directory and move accordingly
        if os.path.isdir(source_path):
            # Move the directory
            shutil.move(source_path, destination_path)
        elif os.path.isfile(source_path):
            # Move the file
            shutil.move(source_path, destination_path)

else:
    print(f"Destination directory {destination_dir} already exists. Skipping download.")
    print("If you want to re-download the data, please delete the destination directory first.")
