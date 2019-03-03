import requests
import time
import re
import traceback
import bs4
import csv
import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy import stats
class Stock:
    def getHTMLText(self,url):
        try:
            r = requests.get(url)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except:
            return ""

    def getStockList(self,stockURL):
        lst=[]
        html = self.getHTMLText(stockURL)
        soup = BeautifulSoup(html, 'html.parser')
        a = soup.find_all('a')
        for i in a:
            try:
                href = i.attrs['href']
                stockCode=(re.findall(r"[s][hz]\d{6}", href)[0])[2:]
                if stockCode is not None:
                    stockName=re.findall(r">(.*)\([\d]{6}\)<",str(i))
                    lst.append(tuple(["".join(stockName),stockCode]))
            except BaseException:
                # exstr = traceback.format_exc()
                # print(exstr)
                continue
        return lst
    def getStockStartDate(self,stockCode):
        html = self.getHTMLText("http://quotes.money.163.com/f10/gszl_" + stockCode + ".html#01f02")
        soup = BeautifulSoup(html, 'html.parser')
        try:
            a = soup.find_all('table', class_="table_bg001 border_box limit_sale table_details")[1]
        except:
            return time.strftime("%Y%m%d")
        cnt = 0;
        date = ""
        for tr in a.children:
            if isinstance(tr, bs4.element.Tag):
                tds = tr('td')
                cnt = cnt + 1
                if (cnt == 2):
                    date = tds[1].string.replace("-", "")
                    break;
        return date

    def getHistoryTradeInfo(self,stockCode):
        download_url = "http://quotes.money.163.com/service/chddata.html?code=0" + stockCode + "&start=" + self.getStockStartDate(
            stockCode) + "&end=" + time.strftime(
            "%Y%m%d") + "&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
        data = requests.get(download_url)
        with open('StockData/history/' + stockCode + '.csv', 'wb') as f:
            for chunk in data.iter_content(chunk_size=10000):
                if chunk:
                    f.write(chunk)
    def writeCvs(self,csvData,path="stockData/stockCode.csv"):
        with open(path,"w+",encoding="UTF-8",newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerows(csvData)
    def readCsv(self,path="stockData/stockCode.csv"):
        birth_data = []
        with open(path) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
                birth_data.append(row)
            return birth_data
    def readStockData(self,dir="stockData/history"):
        filenames=os.listdir(dir)
        preStockDataDict={}
        for filename in filenames:
            stockCode=filename.split(".")[0]
            fullFilePath=os.path.join(dir,filename)
            df = pd.read_csv(fullFilePath,encoding="gbk")
            csv_data=df.values
            header=list(df.columns)
            openPriceIndex=header.index("开盘价")
            closePriceIndex=header.index("收盘价");
            timeIndex=header.index("日期")
            if len(csv_data)>1000:
                print(stockCode)
                openPrice=csv_data[:,openPriceIndex][:1000]
                closePrice=csv_data[:,closePriceIndex][:1000]
                if "None" in openPrice.tolist():
                    continue
                if "None" in closePrice.tolist():
                    continue
                openPriceList=openPrice.tolist()
                closePriceList=closePrice.tolist()
                openPrice=np.array([float(item) for item in openPriceList])
                closePrice=np.array([float(item) for item in closePriceList])
                averagePrice=(openPrice+closePrice)/2
                timeList=csv_data[:,timeIndex][:1000]
                timestamp=[]
                for timeStr in timeList:
                    tmp=time.mktime(time.strptime(timeStr,'%Y-%m-%d'))
                    timestamp.append(tmp)
                tmpDict=dict()
                tmpDict["averagePrice"]=averagePrice
                tmpDict["openPrice"] = openPrice
                tmpDict["closePrice"] = closePrice
                tmpDict["timestamp"]=timestamp
                preStockDataDict[stockCode]=tmpDict
        return preStockDataDict
    def allVisited(self,visited):
        for key,value in visited.items():
            if value==False:
                return False
        return True
    def pearsonCluster(self):
        stockData=self.readStockData()
        visited=dict()
        allClass=[]
        for key in stockData.keys():
            visited[key]=False
        while not self.allVisited(visited):
            for stockCode,stockInfo in stockData.items():
                if not visited[stockCode]:
                    oneClass=[]
                    oneClass.append(stockCode)
                    visited[stockCode]=True
                    for stockCode1,stockInfo1 in stockData.items():
                        if not visited[stockCode1]:
                            r,pValue= stats.pearsonr(stockInfo["averagePrice"],stockInfo1["averagePrice"])
                            if r>0.9:
                                oneClass.append(stockCode1)
                                visited[stockCode1] = True
                    allClass.append(oneClass)
        return allClass

if __name__=="__main__":
    stock=Stock()
    #scratch the stock code from http://quote.eastmoney.com/stocklist.html
    #stockList=stock.getStockList("http://quote.eastmoney.com/stocklist.html")
    #stock.writeCvs(stockList)
    #print(stockList)

    # stockCodeList=stock.readCsv();
    # print(len(stockCodeList))
    #
    # for stockName,stockCode in stockCodeList:
    #     print(stockName+":"+str(stockCode))
    #     stock.getHistoryTradeInfo(str(stockCode))
    # print("success")

    allClass=stock.pearsonCluster()
    stockCodeList=stock.readCsv();
    code2NameMap=dict()
    for stockName,stockCode in stockCodeList:
            code2NameMap[stockCode]=stockName
    allClass=sorted(allClass,reverse=True,key=lambda elem:len(elem))
    for oneClass in allClass:
        nameList=[code2NameMap[item] for item in oneClass]
        print(",".join(nameList))
