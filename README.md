# Discord AI Chat Bot

This is a simple AI chat bot model inspired by https://pytorch.org/tutorials/beginner/chatbot_tutorial.html.

This bot can scan the message history from a discord server, trian a chatbot off that, and then run the chat bot in the discord server.


create your own token.txt containing the token for a discord bot to run this yourself. You will also need to replace the hard-coded ID's set in scanner.py

scanner.py - scan all the messages in a discord server into a bunch of text files in channeltranscripts/

model.py - create or train the chatbot AI model

chat.py - run the trained model in a discord server, responding to messages in a certain channel.