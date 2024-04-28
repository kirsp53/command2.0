import re
import json
import csv
import aiofiles
import aiohttp
from bs4 import BeautifulSoup
import asyncio

async def fetch_and_parse(url):
    links = []
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')


            # Пример поиска элементов на странице
            titles = soup.find_all('a',{"class":"card_full_link"})
            for title in titles:
                href = title.get('href')
                links.append(href)
    return links


async def fetch_and_parse_1(url):
    links = []
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            bad_tags = soup.find("div", {"class": "teachers"})
            if bad_tags:
                bad_tags.clear()
            bad_tags = soup.find("div", {"class": "resume"})
            if bad_tags:
                bad_tags.clear()
            bad_tags = soup.find("div", {"class": "teacher-new__content"})
            if bad_tags:
                bad_tags.clear()
            print(soup)
            title = soup.title.get_text()
            text = soup.get_text()
            return [title,text]

# Пример использования
async def write(textpath,text):
    # Открываем файл для записи
    async with aiofiles.open(textpath,encoding = 'utf-8' ,  mode='w') as file:
        # Записываем данные в файл
        await file.write(text)

async def main():
    url = 'https://gb.ru/courses/all'
    links = await fetch_and_parse(url)
    texts = []
    pattern = r"«(.*?)»"
    negative_pattern = r"с нуля до.*"
    negative_pattern_2 = r": быстрый старт.*"
    with open('../example.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['title', 'clean_title', 'link', 'filepath'])
    for  link in links:

        text = await fetch_and_parse_1(link)
        clean_title = re.findall(pattern, text[0])
        #print(text)
        if len(clean_title) <=0:
            title = text[0]
            #print(title)
        else:
            for t in clean_title:
                pass
                #print(t)
            title = clean_title[0]
        title = re.sub(negative_pattern, "", title)
        clean_title = re.sub(negative_pattern_2, "", title)
        clean_title = re.sub('\n','',clean_title)
        clean_title = clean_title.strip()
        #clean_title = re.sub(' ', '', clean_title)
        clean_text = re.sub(' +', ' ',text[1])
        clean_filename = re.sub(r'[^\w\s]', '_', clean_title)
        clean_text = re.sub('\n+', '\n', clean_text)
        texts.append([text[0],clean_title,clean_text])
        await write("../texts/" + clean_filename+".txt",clean_text)
        data = {
            "title": title,
            "cleantitle": clean_title,
            "link": link,
            "filepath": "../texts/" + clean_filename+".txt"
        }
        with open("courses.json",encoding = 'utf-8',mode= "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
        with open('../example.csv', 'a', newline='') as csvfile:
            # Создаем объект writer
            writer = csv.writer(csvfile)
            writer.writerow([title, clean_title, link,f"texts/{clean_filename}.txt"])

# Запуск асинхронной функции

asyncio.run(main())