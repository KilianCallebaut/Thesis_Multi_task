import os

import paramiko

client = paramiko.SSHClient()
k = paramiko.RSAKey.from_private_key_file(r'C:\Users\mrKC1\.ssh\id_rsa')
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname='145.100.57.141', username='ubuntu', pkey=k)

sftp = client.open_sftp()
# sftp.mkdir('/home/ubuntu/Thesis_Multi_task/data')
# sftp.mkdir('/home/ubuntu/Thesis_Multi_task/data/Data_Readers')
# sftp.mkdir('/home/ubuntu/datasets')

# for root, dirs, files in os.walk('.\data\Data_Readers'):
#     for dir in dirs:
#         print(dir)
#         src = r'./data/Data_Readers/' + dir
#         # tar = os.path.join(r'Thesis_Multi_task', 'data', 'Data_Readers', dir)
#         tar = '/home/ubuntu/Thesis_Multi_task/data/Data_Readers/' + dir
#         print(sftp.listdir('/home/ubuntu/Thesis_Multi_task/data/Data_Readers/'))
#         if dir not in sftp.listdir('/home/ubuntu/Thesis_Multi_task/data/Data_Readers/'):
#             sftp.mkdir(tar)
#         for _, _, f in os.walk(src):
#             for fi in f:
#                 print(fi)
#                 sftp.put(src + "/" + fi, tar + "/" + fi)
remote_base = r'/home/ubuntu/datasets'
local_base = r'F:\Thesis_Datasets\Taken'
for root, dirs, files in os.walk(local_base):
    print("-----------------------")

    levels = root.split(local_base + "\\")[-1] if len(root.split(local_base + "\\")) > 1 else ''
    levels = levels.replace('\\', '/')
    # print(levels)
    # print(remote_base+levels)
    print(root)
    print("levels")
    print(levels)

    remote_dir = remote_base + "/" + levels if levels != '' else remote_base
    print("remote_dir")
    print(remote_dir)
    for dir in dirs:
        print(sftp.listdir(remote_dir))
        if dir not in sftp.listdir(remote_dir):
            print("remote_Dir + dir")
            print(os.path.join(remote_dir, dir))
            sftp.mkdir(os.path.join(remote_dir, dir))

    for f in files:
        print("root+f")
        print(root + "/" + f)
        print("remote_dir+f")
        print(remote_dir + "/" + f)
        sftp.put(root + "\\" + f, remote_dir + "/" + f)

sftp.close()
