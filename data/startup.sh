mkdir /home/ubuntu/big-storage
sudo mkfs -t ext4 /dev/nvme1n1
sudo mount -t ext4 /dev/nvme1n1 /home/ubuntu/big-storage
sudo chown -hR ubuntu /home/ubuntu/big-storage
sudo apt update && sudo apt upgrade -y
pip install torch accelerate matplotlib jupyterlab diffusers ipywidgets git+https://github.com/cloneofsimo/lora.git
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt update && sudo apt install git-lfs -y
cd /home/ubuntu/big-storage
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
