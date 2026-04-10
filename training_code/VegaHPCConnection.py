print("we are at vega")
import time
import pyotp
import paramiko
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class VegaHPCConnection:
    def __init__(self, hostname='_HPC_HOST_', username=None, port=22):
        """
        Initialize Vega HPC connection with 2FA support
        
        Args:
            hostname: Vega HPC hostname
            username: Your username on Vega
            port: SSH port (default 22)
        """
        self.hostname = hostname
        self.username = username 
        self.port = port
        self.ssh_client = None
        self.sftp_client = None
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup connections"""
        self.disconnect()
    
    def disconnect(self):
        """Close SSH and SFTP connections"""
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        logger.info("Disconnected from Vega HPC")
        
    def connect_with_key_and_totp(self,key_file, password=None, totp_code=None):
        """
        Connect to Vega HPC using password + TOTP authentication
        This is the most reliable method for Vega HPC
        """
        try:
            # OTP key placeholder (do not hardcode real TOTP secrets)
            totp = pyotp.TOTP("_OTP_KEY_")
            totp_code= totp.now()
            logger.info(f"Received TOTP code: {len(totp_code)} digits")
            
            private_key = paramiko.Ed25519Key.from_private_key_file(key_file)
            sock = paramiko.Transport((self.hostname, self.port))
            sock.connect(username=self.username,pkey=private_key)
            # handle 2step Verification
            def handler(title, instructions, prompt_list):
                responses = []
                for prompt, show_input in prompt_list:
                    if 'password' in prompt.lower():
                        responses.append(password)
                    elif 'verification code' in prompt.lower() or 'two-factor' in prompt.lower():
                        responses.append(totp_code)
                    else:
                        logger.warning(f"Unexpected prompt: {prompt}")
                        responses.append('')
                return responses
            
            sock.auth_interactive(self.username, handler)

            # Create new SSH client
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client._transport = sock
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            return True

                
        except paramiko.AuthenticationException as e:
            logger.error(f"Authentication failed: {e}")
            logger.error("Common causes:")
            logger.error("1. Incorrect password")
            logger.error("2. Expired or incorrect TOTP code")
            logger.error("3. Time synchronization issue (check your device time)")
            self.ssh_client.close()
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.ssh_client.close()
            import traceback
            logger.error(traceback.format_exc())
            return False
    def submit_comand(self,comand):
            try:

                test_command = "sinfo"  # Show summary of system
                logger.info(f"Executing SLURM  command: {comand}")
            
                stdin, stdout, stderr = self.ssh_client.exec_command(comand)
                output = stdout.read().decode('utf-8').strip()
                error = stderr.read().decode('utf-8').strip()
                if output:
                    logger.info(f"SLURM test successful. Cluster info:")
                    logger.info(f"{output}")
                if error:
                    logger.warning(f"Test command stderr: {error}")
                self.ssh_client.close()
            except Exception as e:
                self.ssh_client.close()
                print(f"An exception occured during command submissoon: {e}")
    def transferFileToHPC(self):
        try:
            logger.info(f"Connecting to {self.hostname} with keyboard-interactive auth...")
            local_file = "_LOCAL_FILE_PATH_"
            remote_file = f"test_transfer_{int(time.time())}.csv"
            def printTotals(transferred, toBeTransferred):
                print (f"Transferred: {transferred}\tOut of: {toBeTransferred}")
        
            # Call the async transfer function
            sftp = self.ssh_client.open_sftp()

            transfer_result = sftp.get()
            
            transfer_result = sftp.put(local_file, remote_file,printTotals)
            
            if transfer_result:
                logger.info("File transfer completed successfully")
            else:
                logger.error("File transfer failed")
            
        except Exception as e:
            self.ssh_client.close()
            logger.error(f"Connection error: {e}")
    def transfer_file_from_HPC(self):
        try:
            logger.info(f"Connecting to {self.hostname} with keyboard-interactive auth...")
            local_folder = "./HPCModels"
            os.makedirs(local_folder, exist_ok=True)

            remote_folder = "MLmodels/"
            def printTotals(transferred, toBeTransferred):
                print (f"Transferred: {transferred}\tOut of: {toBeTransferred}")
            sftp = self.ssh_client.open_sftp()
            remote_files = sftp.listdir(remote_folder)
            # Call the async transfer function
            scalers=[]
            models=[]
            for remote_file_name in remote_files:
                if "scaler" in remote_file_name:
                    scalers.append(remote_file_name)
                else:
                    models.append(remote_file_name)
            scalers.sort()
            models.sort()
            last_model=models[-1]
            last_scaler=scalers[-1]
            transfer=[last_model,last_scaler]
            transferred_count = 0
            failed_transfers = []
            for filename in transfer:
                try:
                    remote_file_path = f"{remote_folder}{filename}"
                    local_file_path = os.path.join(local_folder, filename)
                    
                    logger.info(f"Transferring: {filename}")
                    
                    # Get file info for progress tracking
                    file_attr = sftp.stat(remote_file_path)
                    file_size = file_attr.st_size
                    
                    # Transfer the file
                    sftp.get(remote_file_path, local_file_path, callback=printTotals)
                    
                    transferred_count += 1
                    logger.info(f"Successfully transferred: {filename} ({file_size} bytes)")
                    
                except Exception as file_error:
                    logger.error(f"Failed to transfer {filename}: {file_error}")
                    failed_transfers.append(filename)
            
            sftp.close()
            print("success transfer")
            
        except Exception as e:
            sftp.close()
            self.ssh_client.close()
            logger.error(f"Connection error: {e}")
