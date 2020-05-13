import codecs,numpy,os,random
from matplotlib import pyplot as plt
class xyz:
    data_dict={}
    def __init__(self,sizes):
        self.training_data=None
        self.test_data=None
        self.no_of_layers=len(sizes)
        self.biases=[numpy.random.randn(y,1) for y in sizes[1:]]
        self.weight=[numpy.random.randn(y,x)for x,y in zip(sizes[:-1],sizes[1:])]
    def load_data(self,datapath):
        files=os.listdir(datapath)
        for file in files:
            with open(datapath+'\\'+file,'rb') as f:
                data=f.read()
            types=int(codecs.encode(data[:4],'hex'),16)
            length=int(codecs.encode(data[4:8],'hex'),16)
            if types==2051:
                category='images'
                rows=int(codecs.encode(data[8:12],'hex'),16)
                cols=int(codecs.encode(data[12:16],'hex'),16)
                parser=numpy.frombuffer(data,dtype=numpy.uint8,offset=16)
                parser=parser.reshape(length,rows,cols)
            elif types==2049:
                category='labels'
                parser=numpy.frombuffer(data,dtype=numpy.uint8,offset=8)
            if length==10000:
                sets='test'
            elif length==60000:
                sets='train'
            self.data_dict[sets+' '+category]=parser  
    def load_datawrapper(self):
        training_inputs=[numpy.reshape(x,(784,1)) for x in self.data_dict['train images']]
        training_results=[self.vectorize(y) for y in self.data_dict['train labels'] ]
        self.training_data=list(zip(training_inputs,training_results))
        test_inputs=[numpy.reshape(x,(784,1))for x in self.data_dict['test images']]
        test_results=[self.vectorize(y) for y in self.data_dict['test labels']]
        self.test_data=list(zip(test_inputs,test_results))
    def vectorize(self,y):
        a=numpy.zeros((10,1))
        a[y]=1.0
        return a
    def sigmoid(self,hk):
        return 1.0/(1.0+ numpy.exp(-hk))
    def sigmoid_derivative(self,hk):
        sigma=self.sigmoid(hk)
        return (1-sigma)*sigma
    def cost_derivative(self,y,t):
        return(y-t)
    def feedforward(self,activation):
        for w, b in zip(self.weight, self.biases):
            activation=self.sigmoid(numpy.dot(w,activation)+b)
        return activation
    def evaluate(self):
        test_results=[((numpy.argmax(self.feedforward(img))),numpy.argmax(label))for img,label in self.test_data]
        return sum(int(y==t) for (y,t) in test_results)
            
    def SGD(self,minibatch_size,eta,epochs,IsTesting,x):
        if IsTesting:
            test_len=len(self.test_data)
        train_len=len(self.training_data)
        for i in range(epochs):
            random.shuffle(self.training_data)
            minibatches=[self.training_data[k:k+minibatch_size]for k in range(0,train_len,minibatch_size)]
            for minibatch in minibatches: 
                self.minibatch_update(minibatch,eta)
            if IsTesting:
                #in case testing is to perform varible IsTesting=True
                x.append(self.evaluate())
                print("EPOCH {0}: {1} / {2}".format(i,x[-1],test_len))
            else:
                print("Training Complete Epoch {0}".format(i))
    def minibatch_update(self,minibatch,eta):
        nabla_b=[numpy.zeros(b.shape) for b in self.biases]
        nabla_w=[numpy.zeros(w.shape)for w in self.weight]
        for img,label in minibatch:
           delta_nabla_b,delta_nabla_w=self.backprop(img,label)
           nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
           nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weight=[w-(eta/len(minibatch))*(nw) for w,nw in zip(self.weight,nabla_w)]
        self.biases=[b-(eta/len(minibatch))*(nb) for b,nb in zip(self.biases,nabla_b)]
        
        
    def backprop(self,x,y):
        nabla_b=[numpy.zeros(b.shape) for b in self.biases]
        nabla_w=[numpy.zeros(w.shape) for w in self.weight]
        activation=x
        activations=[x]
        h_list=[]
        #feed forward
        for b ,w in zip(self.biases,self.weight):
            hk=numpy.dot(w,activation)+b
            h_list.append(hk)
            activation=self.sigmoid(hk)
            activations.append(activation)
        # backward pass
        delta=self.cost_derivative(activations[-1],y)*self.sigmoid_derivative(h_list[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=numpy.dot(delta, activations[-2].transpose())
        for l in range(2,self.no_of_layers):
            hj=h_list[-l]
            sp=self.sigmoid_derivative(hj)
            delta=numpy.dot(self.weight[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=numpy.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)
        
if __name__=="__main__":
    x=[]
    datapath=os.getcwd()+'\\mnist'                
    obj=xyz([784,30,10])
    obj.load_data(datapath)
    obj.load_datawrapper()
    obj.SGD(10,0.4,30,True,x)
    y=list(range(30))
    plt.plot(y,x)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    
    
    

