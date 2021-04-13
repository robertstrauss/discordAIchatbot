import discord
import re
import requests
import random
import time
import asyncio
import pytorch as torch
import pandas as pd

client = discord.Client()

guildID = 755473404643115170

guild = None

channelID = 831188849946394654

talkchannel = None

# async def pingean():
#     while True:

#         await asyncio.sleep(pinginterval)
#         print(pinging)
#         if (pinging):
#             role = random.choice(ean.roles)
#             await pingchannel.send('{}'.format(role.mention))


# def startpingean():
#     global pinging
#     pinging = True

# def stoppingean():
#     global pinging
#     pinging = False

data = pd.DataFrame(columns=['content', 'time', 'author'])


@client.event
async def on_ready():
    global guild, pingchannel, ean, logan, sam
    print('We have logged in as {0.user}'.format(client))
    guild = client.get_guild(guildID)
    talkchannel = client.get_channel(pingEanID)

@client.event
async def on_message(message):
    global pinging, pinginterval, pingchannel, ean

    if message.author == client.user:
        return


    if collecting:
        data = data.append({'content': message.content,
                            'time': message.created_at,
                            'author': message.author.name}, ignore_index=True)
    


with open('apikey.txt', 'r') as apikeytxt:
    api_key = apikeytxt.read()
api_url = 'https://www.alphavantage.co/query'
def getQuote(symbol):
    data = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': api_key
    }
    return requests.get(api_url, params=data).json()['Global Quote']



with open('token.txt', 'r') as tokentxt:
    asyncio.get_event_loop().create_task(pingean())
    client.run(tokentxt.read())
