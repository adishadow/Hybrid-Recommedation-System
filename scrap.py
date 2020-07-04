import sys
from bs4 import BeautifulSoup
import requests
import re
import time

def getMovieData(movieid):
  url="https://www.imdb.com/title/tt0"+str(movieid)+'/'
  page = requests.get(url)
  soup = BeautifulSoup(page.content,'html.parser')
  movie_info={}
  movie_info['movie_title']=soup.find('div',class_='title_wrapper').h1.text.split('(')[0]
  movie_info["movie_year"]=soup.find('div',class_='title_wrapper').h1.text
  movie_info['movie_year']=re.sub('[(,)]','',movie_info['movie_year'])
  movie_info['total_review'] = soup.find('div',class_='imdbRating').a.span.text
  movie_info['total_review'] = re.sub('\ {2}','', movie_info['total_review'])
  movie_info['movie_summary'] = soup.find('div',class_='summary_text').text
  movie_info['movie_summary']=re.sub('\ {2,}','',movie_info['movie_summary'])
  movie_info['movie_summary']=re.sub('\n','',movie_info['movie_summary'])
  movie_info['movie_length'] = soup.find('div',class_='title_wrapper')
  movie_info['movie_length']= movie_info['movie_length'].find('div',class_='subtext').time.text
  movie_info['mpvie_length']=re.sub('\ {2}','', movie_info['movie_length'])
  movie_info['movie_length']=re.sub('\n','', movie_info['movie_length'])
  movie_info['movie_dir']=soup.find_all('div',class_="credit_summary_item")
  movie_info['movie_dir']=movie_info['movie_dir'][0].a.text
  movie_writer=soup.find_all('div',class_="credit_summary_item")
  movie_info['writer_list']=[" "," "," "]
  m=movie_writer[1].find_all('a')
  for item in range(len(m)):
    movie_info["writer_list"][item]=m[item].text
    movie_info["writer_list"][item]=re.sub('.*more credit[s]*',' ',m[item].text)
  movie_info['s_list']=[" "," "," "," "," "]
  movie_star=soup.find_all('div',class_="credit_summary_item")
  m=movie_star[2].find_all('a')
  for item in range(len(m)):
    movie_info['s_list'][item]=m[item].text
    movie_info['s_list'][item]=re.sub('See full cast & crew',' ',m[item].text)
  rel_d=soup.find('div',class_='subtext')
  r=rel_d.find_all('a')
  r=(r[-1].text)
  movie_info['r_date']=re.sub('\n','',r)
  movie_info['g_list']=[" "," "," "," "]
  m_genre=soup.find('div',class_='subtext')
  g=m_genre.find_all('a')
  del g[-1]
  for item in range(len(g)):
		  movie_info['g_list'][item]=g[item].text
  movie_info['average_rating']=soup.select(".ratingValue span")[0].text
  movie_info['img'] =soup.find('div', class_='poster')
  movie_info['img'] = movie_info['img'].find_all('img')[0]
  movie_info['img']  = movie_info['img']['src']
  #movie_info['video'] =soup.find('div',class_='slate_wrapper')
  #movie_info['video'] =movie_info['video'].find('div',class_='slate').img['src']
  movie_info['story'] = soup.find('div',class_="inline canwrap").p.span.text
  return movie_info