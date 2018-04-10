from unittest import TestCase
import tool


class TestTool(TestCase):
    LOSS=0.0000001
    def test_calcPearson0(self):
        obj=tool.Tool()
        x=[]
        y=[]
        assert(obj.calcPearson(x,y)==None);
    def test_calcPearson1(self):
        obj=tool.Tool()
        x=[1,2,3]
        y=[4,5,6]
        tmp=obj.calcPearson(x,y)
        assert((obj.calcPearson(x,y)-1<self.LOSS));
    def test_calcPearson2(self):
        obj=tool.Tool()
        x=[1,2]
        y=[4,3.4]
        tmp=obj.calcPearson(x,y)
        assert(tmp-(-1)<self.LOSS);
    def test_calcPearson3(self):
        obj=tool.Tool()
        x=[4,5.3,6]
        y=[4,5,6]
        tmp=obj.calcPearson(x,y)
        assert((tmp-0.98532927816<self.LOSS)==True);
    def test_calcPearson4(self):
        obj=tool.Tool()
        x=[1]
        y=[1]
        tmp=obj.calcPearson(x,y)
        assert(obj.calcPearson(x,y));
    def test_calcPearson5(self):
        obj=tool.Tool()
        x=[0.1,0.1,0.1]
        y=[0.2,0.4,0.5]
        assert((obj.calcPearson(x,y)-0<self.LOSS)==True);

