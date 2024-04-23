## How to start

### 1. Download the data

link: [one drive](https://uniofnottm-my.sharepoint.com/:f:/g/personal/psxah15_nottingham_ac_uk/EoYEHYuEGWVIiJMaIIztgLUB8Ra2_8GR3Nvmnxh_sJDMHw?e=BorlI3)

As you can see, there are two file in the folder. 

- One is `VitaLITy-1.1.0.json`, This is the same as VitaLITy-1.0.0.json in the original vitality, except that it contains ada embedding related information
- The other is `data.db.zip`, which contains information about the chromadb data set. I use this to retrieve data when performing enhancements. After downloading, please unzip it and place the `data.db` folder in the project root folder.

The directory structure is as shown below.

![1.png](./images/1.png)

### 2. Install depedency

just run `pip install -r ./requirements.txt` to install the depedency.

> P.S. You may notice that I changed the items in requirements.txt and removed all versions. That's because when I first developed Vitality, I found that the dependencies could not be installed correctly, so I tried to use each dependency library. The latest version to solve this problem

### 3. Add open ai key to .env

Since the key is sensitive content, I will send it to you privately via email/slack

### 4. start the server

Now you can start the server normally. If you have any questions, you can ask me via email/slack.
