import os

import paramiko

client = paramiko.SSHClient()
k = paramiko.RSAKey.from_private_key_file(r'C:\Users\mrKC1\.ssh\id_rsa')
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='145.100.57.141', username='ubuntu', pkey=k)

sftp = client.open_sftp()
# sftp.mkdir('/home/ubuntu/Thesis_Multi_task/data')
# sftp.mkdir('/home/ubuntu/Thesis_Multi_task/data/Data_Readers')

for root, dirs, files in os.walk('.\data\Data_Readers'):
    for dir in dirs:
        print(dir)
        src = r'./data/Data_Readers/' + dir
        # tar = os.path.join(r'Thesis_Multi_task', 'data', 'Data_Readers', dir)
        tar = '/home/ubuntu/Thesis_Multi_task/data/Data_Readers/'+dir
        sftp.mkdir(tar)
        for _, _, f in os.walk(src):
            for fi in f:
                sftp.put(src + '/' + fi, tar + '/' + fi)
sftp.close()
