# British Isles pixel map — Python version, works in Google Colab.
# Paste this whole thing into one cell and run it.

from collections import deque

import numpy as np
import matplotlib.pyplot as plt

# THE PIXELS. One character per pixel.
#   .    =  sea (left blank)
#   1-9  =  London heat: 9 is the very center, 1 the outer edge
#   any letter  =  ordinary land

pixels = [
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
".......................................................................................................................................................",
"............................................................................................S..........................................................",
"...........................................................................................SSS.....S...................................................",
"............................................................................................SSS..SSS...................................................",
"...............................................................................................SSS.....................................................",
"...........................................................................................SS.SS.......................................................",
"........................................................................................SSSSS..S.S.....................................................",
"........................................................................................SSSSS...SSS....................................................",
"........................................................................................SSSS...........................................................",
"........................................................................................SSSSSSS........................................................",
"........................................................................................SSSSSSSS.......................................................",
".......................................................................................SSS...SSS.......................................................",
".......................................................................................SSS...S.........................................................",
"........................................................................................SSS.SS.........................................................",
".........................................................................................S..SS.........................................................",
".......................................................................................................................................................",
".......................................................................................SSS.............................................................",
"....................................................................SS..............SSSSSSSS...........................................................",
"...................................................................SSSSS.SS...SSSSSSSSSSSSSS...........................................................",
"..................................................................SSSSSSSSSSSSSSSSSSSSSSSSS............................................................",
"..................................................................SSSSSSSSSSSSSSSSSSSSSSSSSS.........................................................",
"..................................................................SSSSSSSSSSSSSSSSSSSSSSSSSS.........................................................",
"..................................................................SSSSSSSSSSSSSSSSSSSSSSSSSS.........................................................",
"..............................................SSSSSSSS...........SSSSSSSSSSSSSSSSSSSSSSSSSS............................................................",
".............................................SSSSSSSS.............SSSSSSSSSSSSSSSSSSSSSSSS.............................................................",
"............................................SSSSSSSS.S.........SSSSSSSSSSSSSSSSSSSSSSSSS...............................................................",
"..........................................SSSSSSSSSSSS.........SSSSSSSSSSSSSSSSSSSSSSSS................................................................",
"..........................................SSSSSSSSS.............SSSSSSSSSSSSSSSSSSSSSS.................................................................",
".........................................SSSSSSSSSS.............SSSSSSSSSSSSSSSSSSSSS..................................................................",
"..........................................SSSSSSSSS...........SSSSSSSSSSSSSSSSSSSSSS...................................................................",
"..........................................SSSSSSSSS............SSSSSSSSSSSSSSSSSSSS....................................................................",
"..........................................SSSSSSSS..............SSSSSSSSSSSSSSSSSS.....................................................................",
"............................................SSSSSS..........S..S.SSSSSSSSSSSSSSSS......................................................................",
"..........................................SSSSSS..........SSSSSSSSSSSSSSSSSSSSSS.......................................................................",
".......................S.................SSSSSS...........SSSSSSSSSSSSSSSSSSSSSSS.S....................................................................",
"..........................................SSS.............SSSSSSSSSSSSSSSSSSSSSSSSS....................................................................",
"...........................................SS.............SSSSSSSSSSSSSSSSSSSSSSSS.....................................................................",
"........................................S..........S......SSSSSSSSSSSSSSSSSSSSSSS.....SSSSS..SSSSS...S.SS..............................................",
".....................................SSSSSS.......SSS.....SSSSSSSSSSSSSSSSSSSSSS...SSSSSSSSSSSSSSSSSSSSSSSS............................................",
"....................................SSSSSS.....S..SSSS....SSSSSSSSSSSSSSSSSSSSS.S.SSSSSSSSSSSSSSSSSSSSSSSSS............................................",
".....................................SSSSS.....SS.SSSS.S.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS............................................",
".......................................SSS....S.SSSSSS.S.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...........................................",
"......................................SSS.....SSSSSSSS.S.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...........................................",
"......................................SSS.....SSSSSSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS............................................",
"......................................SS......SSSSSSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS............................................",
"......................................SSS.......S.SSSSSS...SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.............................................",
"......................................SSS........SSSSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS..............................................",
".....................................SSS..........SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS..............................................",
"......................................SS...........SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...............................................",
"......................................SS..............S.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...............................................",
"......................................SSS..............SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...............................................",
".....................................S..........SSSS...SS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...............................................",
".....................................SS...........SSS....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS................................................",
"....................................SS............SSS....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS................................................",
".........................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS................................................",
"..........................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.................................................",
".........................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.................................................",
".......................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS..................................................",
"....................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...................................................",
".................................................S...SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...................................................",
"................................................SS...SSSSSSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS....................................................",
"...............................................SS..SSSSSSSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS....................................................",
".............................................SS....SSSSS.SSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.....................................................",
"...........................................SSS......SSSSSSSS..SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS......................................................",
"...........................................SS........SSSSSSS..SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS........................................................",
".....................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS........................................................",
".....................................................SSSSSS..SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS........................................................",
"...................................................SSSSSSS..SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS......................................................",
"...................................................SSS......SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.....................................................",
"...........................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS......................................................",
"..........................................................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.S........................................................",
".....................................................S....SS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...........................................................",
"....................................................SS...SS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS............................................................",
"....................................................SS..SSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSS....SSSS.....................................................",
".......................................................SSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSS....SSSSSSS...................................................",
".......................................................SSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS..................................................",
"....................................................SSSSS..SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...............................................",
"..................................................SSSSSSS..SSSSSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...............................................",
".................................................SSSSSSS...SSSSS..SS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS..............................................",
".................................................SSSSSS.....SSSS..SS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEE.............................................",
".................................................SSSSSSS....SSS.SS...SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEE.............................................",
".................................................S..SSS....SSS.SSS...SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEE............................................",
"...................................................SSS.....SSS.SSSS...SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEE..........................................",
"...................................................SS......SSS.SSSS....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEE.........................................",
"...........................................................SSS.SSSS.....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEE.........................................",
"...........................................................SSS.SSSS.....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEE.........................................",
"..........................................................SSS...SSS....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEE.........................................",
"......................................I...................SSSS........SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEE.........................................",
"......................................III.................SSSS........SSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEE.........................................",
".....................................IIIII..........NN....SSS........SSSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEE........................................",
".................................IIIIIIIIIII.....N...................SSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEE........................................",
"..............................IIIIIIIIIIIIII...NNNNNNNN..............SSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEE........................................",
"...........................IIIIIIIIIIIIIII.NNNNNNNNNNNN.............SSSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEE........................................",
"...........................IIIIIIIIIIIIII..NNNNNNNNNNNN............SSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEE........................................",
"...........................IIIIIIIIIIIIIN..NNNNNNNNNNNNN...........SSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEEE.......................................",
"..........................IIIIIIIIIIIIINNNNNNNNNNNNNNNNN..........SSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEEEEE.......................................",
"........................IIIIIIIIIIIIIINNNNNNNNNNNNNNNNNNN........SSSSSSSSSSSSSSSSSSSSSSSSSSSEEEEEEEEEEEEEEEEEEEEE......................................",
".........................IIIIIIIIIIIIINNNNNNNNNNNNNNNNNNN........SSSSSSSSSSSSSSSSSSSSS..EEEEEEEEEEEEEEEEEEEEEEEEE......................................",
".........................IIIIIIIIIIIIINNNNNNNNNNNNNNNNNNNN.......SSSSSSSSSSSSSSSSSSSSS.EEEEEEEEEEEEEEEEEEEEEEEEEE......................................",
"........................IIIIIIIIIIIIIINNNNNNNNNNNNNNNNNNNNNN.......SSS.SSSSSSSSSSSSSS....EEEEEEEEEEEEEEEEEEEEEEEEEE......................................",
"........................IIIIIIIIIIIIIINNNNNNNNNNNNNNNNNNNNNNN.......SS..SSSSS..SSSS.....EEEEEEEEEEEEEEEEEEEEEEEEEEE.....................................",
"......................IIIIIIIIIIIIIINNNNNNNNNNNNNNNNNNNNNNNN........S....SSS..........EEEEEEEEEEEEEEEEEEEEEEEEEEEE.....................................",
".....................IIIIIIIIIIINNNNNNNNNNNNNNNNNNNNNNNNNN..........SS....SS..........EEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................................",
".....................IIIIIIIIIIINNNNNNNNNNNNNNNNNNNNNNNNNNNNNN......SS...............EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................................",
"......................IIIIIIIIIIIINNNNNNNNNNNNNNNNNNNNNNNNNNNN.......................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.................................",
".........................I..IIIIINNNNNNNNNNNNNNNNNNNNNNNNNNNNN.......................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...............................",
"...........................IIINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN.....................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.............................",
".........................IIIINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN.NN......................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE............................",
"........................IIIIINNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN............M..........EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE............................",
".......................IIIIIIINNNNNNNNNNNNIINNNNNNNNNNNNNNNNNN...........MMM..........EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...........................",
"........II.............IIIIIIIINNNNNNNNNNIIIINNNNNNNNNNNNNNNNN...........MMM...........EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........................",
".....II.IIIIIIII..II...IIIIIIIIINNNNNNNNNIIIINNNNNNNNNNNNNNNN...........MMMMM..........EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........................",
".....IIIIIIIIIII.IIIIIIIIIIIIIIIINNNNNNNNIIIINNNNNNNNNNNNNNN............MMMM............EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.........................",
".....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIINNNNNIIIIIINNNNNNNNNNNN.............MMMMM............EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.........................",
".....I.IIIIIIIIIIIIIIIIIIIIIIIIIIIIIINNNNIIIIIIIINNNNNNNNN..............MMMM..............EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.......................",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIINNNNINNNNN.............MMMM...............EE...EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE......................",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIINIIIIINN..............M.M....................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE........................",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE........................",
"...IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..I.......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.......................",
".....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.........................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.......................",
".......II..IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.......................",
"...........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE......................",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.....................",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.....................",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................",
"......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.....................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...E...................",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII....................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.....................",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII....................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII......................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...................",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..................WWWW.............EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..................",
"......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..................WWWW...........W.EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..................",
".......III.IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.................WWWWWWWWW..WW..WWWW.EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..................",
".........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.................WWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.................",
"..........I.II....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..................WWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.................",
".................IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII...................WWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.................",
".........II....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII....................WWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.................",
"..........I...IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII....................WWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.................",
"..............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII....................WWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..................",
".............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..................WWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.....E.............",
".............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.................WWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....EEEEEEEEEE......",
".............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.................WWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....EEEEEEEEEEEE....",
".............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII................WWWW...WWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..EEEEEEEEEEEEEE...",
"............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.................WWW....WWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..",
"............IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII........................WWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.",
"..........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.........................WWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
"..........IIIIIIII.IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..........................WWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
".........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.........................WWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.........................WWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
"..........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..........................WWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
"..........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..........................WWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII...........................WWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
".......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII............................WWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
"........IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII...........................WWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
".....I..IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII...........................WWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.",
"..IIIII.IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.........................WWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.",
".IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.......................WWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.",
"IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.............................WWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.",
"IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.I............................WWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..",
"......IIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.................................WWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..",
"....IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..............................WWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...",
"...IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..............................WWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....",
".IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII..............................WWWWWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.....",
".IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII...............................WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE......",
".IIIIIIIIIIIIIIIIIIIIIIIIII.III...................................WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE......",
"..I.IIIIIIIIIIIIIIIIIIIIIIIIII....................................WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.EEE......",
"....IIIIIIIIIIIIIIIIIIIIIII......................................WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........",
"......IIIIII.IIIIIIIIIIIIII......................................WWWWWWW...WWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........",
".....IIIII.IIIIIIIIIIIIII.........................................WWWWWW......WWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........",
"....IIIIIIIIIIIIIIIIIII............................................WW.......WWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........",
"........IIIIIIIIIIII........................................................WWWWW..WWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........",
".........IIIIIIII..................................................................WWWWWWWWW....EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE............",
"........II....I....................................................................WWWWWWWW...EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.............",
".....................................................................................WWWWWW...EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE..........",
"......................................................................................WWWW...EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.EEEEE....",
"............................................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....",
"............................................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....",
".................................................................................EEEEE......EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....",
".............................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....",
".............................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....",
".............................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.....",
".............................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE........",
".........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.........",
".........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.........",
".........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.........",
".........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.............",
".........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...............",
".........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.EEEEEEE.......EEEEE.................",
"........................................................................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE...E............E..................",
".......................................................................EEEEEEEEEEEEEEEEEEEEE....EEEEEEEEEE....EEEEEEE..................................",
"......................................................................EEEEEEEEEEEEEEEEEEE........EEEEEEEEE.....EEEEE...................................",
".....................................................................EEEEEEEEEEEEEEEEEE...........E....EE.......EEE....................................",
"...................................................................EEEEEEEEEEEEEEEEEEEE............E...................................................",
"...................................................................EEEEEEEEEEEEEEEEEEEE................................................................",
"...................................................................EEEEEEEEEEEEEEEEEEEE................................................................",
"..................................................................EEEEEEEEEEEEEEEEEEEEE................................................................",
"..................................................................EEEEEEEEEEEEEEEEEEEE.................................................................",
".................................................................EEEEEEEEE...E.EEEEEE..................................................................",
"................................................................EEEEEEE..........EEEE..................................................................",
".............................................................E.EEEEEEEE...........EEE..................................................................",
"...........................................................EEEEEEEEEE..................................................................................",
"...........................................................EEEEEEEE....................................................................................",
"...........................................................EEE.EEEE....................................................................................",
"...........................................................EE...EEE....................................................................................",
"................................................................EE....................................................................................."
]

# THE DRAWING.
# Build a picture the same size as the grid and color it in,
# one pixel at a time, based on what character sits there.

SEA  = (247, 245, 240)   # background
LAND = (166, 166, 166)   # grey for all land

heat = {                  # yellow at the edge -> deep red at the center
    "1": (248, 217,  78),
    "2": (249, 195,  63),
    "3": (249, 170,  51),
    "4": (246, 143,  42),
    "5": (239, 114,  34),
    "6": (226,  85,  28),
    "7": (207,  58,  22),
    "8": (181,  36,  16),
    "9": (143,  13,   6),
}

height = len(pixels)
width  = max(len(row) for row in pixels)   # rows aren't all the same length, so size to the longest
is_land = np.zeros((height, width), dtype=bool)

for row, line in enumerate(pixels):
    for col, ch in enumerate(line):
        if ch != ".":
            is_land[row, col] = True

# close stray one-pixel sea holes inside the landmass: flood-fill sea from
# the outer border, then anything left unreached is an enclosed hole, not
# real coastline, so treat it as land
from collections import deque
reached = np.zeros((height, width), dtype=bool)
q = deque()
for r in range(height):
    for c in (0, width - 1):
        if not is_land[r, c] and not reached[r, c]:
            reached[r, c] = True
            q.append((r, c))
for c in range(width):
    for r in (0, height - 1):
        if not is_land[r, c] and not reached[r, c]:
            reached[r, c] = True
            q.append((r, c))
while q:
    r, c = q.popleft()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not is_land[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))
enclosed_hole = (~is_land) & (~reached)

picture = np.full((height, width, 3), SEA, dtype=np.uint8)
for row, line in enumerate(pixels):
    for col, ch in enumerate(line):
        if ch in heat:
            picture[row, col] = heat[ch]
        elif ch != ".":
            picture[row, col] = LAND
picture[enclosed_hole] = LAND

# crop tight around the landmass (plus a small margin) so it's centered
# in the frame instead of floating in a lot of empty sea
land_rows, land_cols = np.nonzero(is_land | enclosed_hole)
margin = 3
r0, r1 = max(land_rows.min() - margin, 0), min(land_rows.max() + margin + 1, height)
c0, c1 = max(land_cols.min() - margin, 0), min(land_cols.max() + margin + 1, width)
picture = picture[r0:r1, c0:c1]
height, width = picture.shape[:2]

plt.figure(figsize=(7, 11))
plt.imshow(picture)
plt.axis("off")
plt.show()

# build a clean, uniform character grid that matches the cleaned-up,
# cropped `picture` array exactly (holes closed, same crop window),
# so the HTML/canvas version matches the PNG instead of drifting from it
char_grid = [["."] * (c1 - c0) for _ in range(r0, r1)]
for out_r, row in enumerate(range(r0, r1)):
    line = pixels[row] if row < len(pixels) else ""
    for out_c, col in enumerate(range(c0, c1)):
        ch = line[col] if col < len(line) else "."
        if enclosed_hole[row, col]:
            ch = "X"  # closed hole -> plain land
        char_grid[out_r][out_c] = ch
cropped_pixels = ["".join(row) for row in char_grid]

CELL = 3  # pixels per grid cell, smaller = smaller page footprint

html = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin: 0; height: 100%%; overflow: hidden; }
  body { display: flex; justify-content: center; align-items: center; background: #f7f5f0; }
  canvas { max-width: 95vw; max-height: 95vh; width: auto; height: auto; display: block; }
</style>
</head>
<body>
<canvas id="c" width="%d" height="%d"></canvas>
<script>
const pixels = %s;
const CELL = %d;
const LAND = "#a6a6a6";
const heat = [null,"#f8d94e","#f9c33f","#f9aa33","#f68f2a",
  "#ef7222","#e2551c","#cf3a16","#b52410","#8f0d06"];
const ctx = document.getElementById("c").getContext("2d");
for (let row = 0; row < pixels.length; row++)
  for (let col = 0; col < pixels[row].length; col++) {
    const ch = pixels[row][col];
    if (ch === ".") continue;
    ctx.fillStyle = (ch >= "1" && ch <= "9") ? heat[Number(ch)] : LAND;
    ctx.fillRect(col * CELL, row * CELL, CELL - 0.6, CELL - 0.6);
  }
</script></body></html>""" % (width * CELL, height * CELL, str(cropped_pixels), CELL)

with open("map.html", "w") as f:
    f.write(html)

print("Saved map.html next to this script.")