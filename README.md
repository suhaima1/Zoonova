# Project Name

The project name is **Zoonova Chatbot**.

## Features
- try catch mechanism to enhance the code quality.
- Seamless deployment on AWS.
- User-friendly web interface using Streamlit.
- Easy setup and installation with the provided `myenv` file.
- Simple integration into existing web pages using the `index.html` file.
- A python script that runs on the console and print the results.

## Installation

To run the Zoonova Chatbot, follow these steps:

1. Unzip the deliverable.zip to your local machine.
2. Navigate to the project directory "deliverable/".
3. Set up a virtual environment using the provided `myenv` file:

```bash
$ source myenv/bin/activate  # for Mac/Linux
$ .\myenv\Scripts\activate  # for Windows
```

4. Install the necessary dependencies by running:

```bash
$ pip install -r requirements.txt
```

## Usage

To run the Zoonova Chatbot, execute the following command in your command prompt or terminal:

```bash: to run the streamlit app
## navigate to streamlit app directory manually or using the following command
$ cd streamlit_app
$ streamlit run zoonova_chatbot.py

```
After running the command, the Streamlit app will be launched, and you can start interacting the provided user interface.

```bash to run the script in terminal
## navigate to python script directory manually or using the following command
$ cd python_script
$ python script.py
```
After running the command, you will be prompted to enter the query.

## Deployment

Zoonova Chatbot is currently deployed on AWS EC2.
To embed the chatbot in your web page using an iframe. Update the iframe source with the AWS endpoint URL. see `index.html` example.

```html
<iframe src="AWS_ENDPOINT_URL" width="100%" height="700"></iframe>
```

Replace `AWS_ENDPOINT_URL` with the provided endpoint URL.
