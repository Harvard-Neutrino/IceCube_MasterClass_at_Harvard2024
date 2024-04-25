# standard methods
from pandas import read_parquet, concat


# custom methods
from .event_reader import Event
from .plot_event import *

# torch methods
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MLEventSelection():

    def __init__( self, path , N=None, min_hits = 10):

        out = read_parquet( path )

        hits_flag = [len(out["photons"][out["photons"].index[i]]["sensor_pos_x"]) > min_hits for i in range(len(out["photons"]))]
        out = out[hits_flag]

        if N is not None:
            out = out[:N]


        self.N_events = len( out["photons"] )

        self.event_hit_info = out["photons"]
        self.mc_truth = out["mc_truth"]

        return None
    
    def __getitem__(self, idx):
        if (idx < 0) or (idx > self.N_events):
            raise IndexError
        return Event( 
            self.event_hit_info[self.event_hit_info.index[idx]], 
            self.mc_truth[self.mc_truth.index[idx]]
        )

class Net(nn.Module):

    def __init__(self, width=1000):
        super(Net, self).__init__()
        # define two convolutional layers
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        # define three linear layers
        self.fc1 = nn.Linear(1944, width)  # 3888 from image dimension
        self.fc2 = nn.Linear(width, 1)

    def forward(self, input):
        # Convolution layer C1: 2 input image channels, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 14, 14) Tensor
        s2 = F.max_pool2d(c1, 2)
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        # f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = (F.tanh(self.fc2(f5)) + 1)/2.
        #output = torch.where(output<0,0,output)
        #output = torch.where(output>1,1,output)
        return output
    
class CustomImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        #self.labels = np.expand_dims(np.cos(self.labels[:,0]),-1)
        self.labels = np.expand_dims(self.labels[:,0],-1)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


    
class MLHelper():

    def generate_dataset(self,mu_events,e_events,N,m=87):
        # function to generate the input images to the CNN
        images = np.empty((2*N,2,m,m),dtype=np.float32)
        labels = np.empty((2*N,4),dtype=np.float32)
        self.hits = concat([mu_events.event_hit_info,
                           e_events.event_hit_info])
        self.mc_truth = concat([mu_events.mc_truth,
                               e_events.mc_truth])
        for i,(event,truth) in enumerate(zip(mu_events.event_hit_info,
                                             mu_events.mc_truth)):
            bins = np.linspace(-0.5,m-0.5,m+1)
            nhits,_,_ = np.histogram2d(event["string_id"],event["sensor_id"],bins=bins)
            first_hit = np.zeros((m,m))
            for j,hit_t in enumerate(event["t"]):
                if first_hit[event["string_id"][j],event["sensor_id"][j]]==0 or first_hit[event["string_id"][j],event["sensor_id"][j]] > hit_t:
                   first_hit[event["string_id"][j],event["sensor_id"][j]] = hit_t
            images[i,0,:,:] = nhits
            images[i,1,:,:] = first_hit
            
            # label
            labels[i] = [0,truth["final_state_energy"][0],truth["final_state_zenith"][0],truth["final_state_azimuth"][0]]
        for i,(event,truth) in enumerate(zip(e_events.event_hit_info,
                                             e_events.mc_truth)):
            bins = np.linspace(-0.5,m-0.5,m+1)
            nhits,_,_ = np.histogram2d(event["string_id"],event["sensor_id"],bins=bins)
            first_hit = np.zeros((m,m))
            for j,hit_t in enumerate(event["t"]):
                if first_hit[event["string_id"][j],event["sensor_id"][j]]==0 or first_hit[event["string_id"][j],event["sensor_id"][j]] > hit_t:
                    first_hit[event["string_id"][j],event["sensor_id"][j]] = hit_t
            images[i+N,0,:,:] = nhits
            images[i+N,1,:,:] = first_hit/4000.
            
            # label
            labels[i+N] = [1,truth["final_state_energy"][0],truth["final_state_zenith"][0],truth["final_state_azimuth"][0]]
        randperm = torch.randperm(len(images))
        self.hits = self.hits.iloc[np.array(randperm)].reset_index(drop=True)
        self.mc_truth = self.mc_truth.iloc[np.array(randperm)].reset_index(drop=True)
        return torch.from_numpy(images[randperm]),torch.from_numpy(labels[randperm])

    def __init__(self, mu_path, e_path, N=2000):
        
        # make our event files
        self.mu_events = MLEventSelection(mu_path,N=N)
        self.e_events = MLEventSelection(e_path,N=N)
        self.N = N

        
    def MakeTrainingDataset(self, N_train, batch_size=32):
        
        self.N_train = N_train
        self.N_test = 2*self.N - N_train

        # make sure we have enough training events
        assert(N_train < 2*self.N)

        # load the train and test images and labels
        self.images, self.labels = self.generate_dataset(self.mu_events,self.e_events,N=self.N)

        # turn these into pytorch dataloaders
        self.train_dataloader = DataLoader(CustomImageDataset(self.images[:N_train],self.labels[:N_train]), batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(CustomImageDataset(self.images[-self.N_test:],self.labels[-self.N_test:]), batch_size=batch_size, shuffle=True)

        
    def MakeNetwork(self,width=1000,lr=0.001,momentum=0.9,gamma=0.9):
        # define the CNN
        self.net = Net(width=width)
        print("Created neural network with %d trainable parameters"%sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self.criterion = nn.functional.binary_cross_entropy

    def train(self, num_epochs=10):
        loss_dict = {}
        for epoch in range(num_epochs):
            print("Beginning epoch %d/%d"%(epoch+1,num_epochs))
            loss_dict[epoch] = []
            for input,target in self.train_dataloader:
                self.optimizer.zero_grad()   # zero the gradient buffers
                output = self.net(input)
                loss = self.criterion(output, target)
                print("Training loss: %2.3f"%loss,end="\r")
                loss_dict[epoch].append(loss.detach().numpy())
                loss.backward()
                self.optimizer.step()    # Does the update
            print ("\033[A                            \033[A")
            self.scheduler.step()
        return loss_dict
    
    def plot_event(self, idx, reveal_network_predition=True, reveal_true_label=True, testset=True):

        if testset:
            idx += self.N_train
        
        if idx > len(self.hits):
            print("Index %d is out of bounds! Try again"%idx)
            return

        event = Event(self.hits[idx],self.mc_truth[idx])

        self.net.eval()
        input = torch.from_numpy(np.expand_dims(self.images[idx],0))
        output = self.net(input).detach().numpy().item()
        label = self.labels[idx].detach().numpy()[0]
        
        layout = get_3d_layout()
        plot_det = plot_I3det()

        fig = go.FigureWidget(data=plot_det, layout=layout)

        plot_evt = plot_first_hits(event)
        fig.add_trace(plot_evt)

        if reveal_network_predition:
            fig.add_annotation(x=0.5, y=1.0,
                text="Network Electron Score: %2.2f"%output,
                showarrow=False,
                )
        if reveal_true_label:
            fig.add_annotation(x=0.4 if label==0 else 0.42, y=0.9,
                text="True Label: %s"%("Muon" if label==0 else "Electron"),
                showarrow=False,
                )

        fig.show()

