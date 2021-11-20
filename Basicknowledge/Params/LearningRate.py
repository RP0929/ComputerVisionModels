def demo():
   import random
   a=0.01
   x=random.randint(1,10)
   y = x * x + 2
   index=1
   while index < 1000 and abs(y-2) > 0.01 :
       y=x*x+2
       print("batch={} x={} y={}".format(index,x,y))
       x=x-2*x*a
       index+=1
demo()