import json
import pandas as pd
import requests
import pickle
import os
import time
# vlist=['command_injection','open_redirect','path_disclosure','remote_code_execution','sql','xsrf','xss']
vlist=['xss']
# vtype='command_injection'
for vtype in vlist:
# file_path = "../python_data/"+vtype
    file_path = os.path.join("../python_data/","plain_"+vtype)
    # file_path = os.path.join("/user/home/ud21703/codebert/python_data/python_data/","plain_"+vtype)
    df = pd.DataFrame(columns=['id','code','label'])

    with open(file_path) as f:
        for line in f:
            js=json.loads(line.strip())
            for i in js:
                for j in js[i]:
                    # time.sleep(1.44)
                    data = js[i][j]
                    commiturl=data['url']
                    response = requests.get(commiturl)
                    if response.ok:

                        response_json=response.json()
                        r_files = response_json['files']
                        neg_text=''
                        for j_file in r_files:
                            filename=j_file['filename']
                            if filename[-3:] =='.py':
                                raw_url = j_file['raw_url']
                                raw = requests.get(raw_url)
                                if raw.ok:
                                    neg_text+=filename
                                    neg_text+='/n/n'
                                    neg_text+=raw.text
                                    neg_text+='/n/n/n'

                        dic={'id':j,'code':neg_text,'label':0}
                        df = df.append(dic,ignore_index=True)

                        files = data['files']
                        pos_text=''
                        for filename in files:
                            file=files[filename]
                            if filename[-3:] =='.py' and 'sourceWithComments' in file.keys():
                                pos_text += filename
                                pos_text += '/n/n'
                                pos_text += file['sourceWithComments']
                                pos_text += '/n/n/n'
                        dic={'id':j,'code':pos_text,'label':1}
                        df=df.append(dic,ignore_index=True)

    # df.pickle(open(os.path.join("../python_data",vtype+"_df.bin"),'wb'))
    # pickle.dump(df,open(os.path.join("/user/home/ud21703/codebert/python_data/python_data/",vtype+"_df.bin"),'wb'))
    # df.to_csv(os.path.join("/user/home/ud21703/codebert/python_data/python_data/",vtype+"_df.csv"))
    df.to_csv(os.path.join("../python_data/",vtype+"_df.csv"))

    jsonfile=df.to_json(orient="records")
    parsed = json.loads(jsonfile)
    jsdp=json.dumps(parsed)
    with open(os.path.join("../python_data/",vtype+".json"),"w") as f:
        f.write(jsdp)
        f.close()
                    





