#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 03:10:27 2018

@author: ryanday

Read symmetry operator list and generate the transformation matrices
from format (a*X+b,a'*Y+b',a"*Z+b") etc. In principle, user should pass 
"""

from lxml import html
import requests


if __name__=="__main__":
    
    base_name = "http://www.globalsino.com/EM/page"
    for i in range(1):
        print(i)
        filename = base_name + "{:d}.html".format(1480+i)
        page = requests.get(filename)
        tree = html.fromstring(page.content)
        title = tree.xpath('//span[@class="STYLE2"]/text()')
        ops = tree.xpath('//div[@class="STYLE19"]/text()')
        if len(title)>0:
            tmp = title[0].split('      ')
            if tmp[-1]=="Space Group":
                print(1400+i,tmp[0])
        print(ops)

#
#<div align="center"><span class="STYLE2">I4/mmm (139) Space Group <br>
#<HTML><HEAD><TITLE>I4/mmm (139) space group</TITLE>

