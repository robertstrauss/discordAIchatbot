import discord
import re
import sys, os
import requests
import random
import time
import asyncio
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




@client.event
async def on_ready():
    global guild, pingchannel, ean, logan, sam
    print('We have logged in as {0.user}'.format(client))
    guild = client.get_guild(guildID)
    # talkchannel = client.get_channel(channelID)

    limit = int(re.findall(r'limit=(\d+)', " ".join(sys.argv))[0])

    channelIDs = re.findall(r' (\d+) ', " ".join(sys.argv))

    if len(channelIDs) == 0:
        channelIDs = [channel.id for channel in guild.channels]

    for channelID in channelIDs:
        channelID = int(channelID)
        channel = client.get_channel(channelID)

        if type(channel) == discord.channel.TextChannel:
            print('scanning', channel)
            data = await collectmessages(channel, limit)
            
            category = guild.get_channel(channel.category_id)
            if type(category) == discord.channel.CategoryChannel:
                folder = 'channeltranscripts/{}'.format(category.name)
            else:
                folder = 'channeltranscripts'
            
            os.makedirs(folder, exist_ok=True)
            print(data)
            # data.to_csv('{}/{}.csv'.format(folder, channel.name))


async def collectmessages(channel, limit):
    data = pd.DataFrame(columns=['content', 'time', 'author'])

    try:
        async for i, msg in enumerate(channel.history(limit=limit)):
            if msg.author != client.user:
                content = msg.content.replace('\n', ' ')

                if msg.author == data[-1].author:
                    data[-1].content += content
                else:
                    data = data.append({'content':  content,
                                        # 'time':     msg.created_at,
                                        'author':   msg.author.name},
                                        ignore_index=True)
            if len(data) == limit:
                break
    except (discord.errors.Forbidden):
        print('access denied to', channel.name)
    finally:
        return data




with open('token.txt', 'r') as tokentxt:
    # asyncio.get_event_loop().create_task(pingean())
    client.run(tokentxt.read())
