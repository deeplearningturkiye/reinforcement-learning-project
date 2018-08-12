
import numpy as np
import random

class Env():
    def __init__(self,random_play):
        self.random_play=random_play
        self.state=[[0,0,0],[0,0,0],[0,0,0]]
        self.reward=0.5
        self.done=0
        self.random_agent_value="O"
        self.winner="Random Agent"
        
    def step(self,action,agent_value):
        ## action is a array like [0,1] state

        if(agent_value=="X" and self.control_win(self.state)==self.random_agent_value):
            self.reward=0
            self.done=1


        
        if(self.state[action[0]][action[1]]!=0):
            print("there is something wrong    :   this state is not avaliable for playing.")
        else:
            self.state[action[0]][action[1]]=agent_value

        
            
        if(self.control_win(self.state)==agent_value):
            self.reward=1
            self.done=1
            self.winner="agent X"
        if(self.control_win(self.state)=="DRAW"):
            self.reward=0
            self.done=1
            self.winner = "DRAW"
        if(self.done!=1):
            if(self.random_play==1):
                self.random_agent_act()
        return(self.state,self.reward,self.done,self.winner)
            
    def control_win(self,state):
        done=0
        for i in range(3):
            if state[i][0] != 0 and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
                done=1
                return state[i][0]
            if state[0][i] != 0 and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
                done=1
                return state[0][i]
            if state[0][0] != 0 and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
                done=1
                return state[0][0]
            if state[0][2] != 0 and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
                done=1
                return state[0][2]
        empty_spaces=[]
        values_of_empty_spaces=[]
        
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    empty_spaces.append([i,j])
                    values_of_empty_spaces.append(0)
        if (len(values_of_empty_spaces)==0):
            return "DRAW"
        else:
            return 0
                    
                
    def random_agent_act(self):
        empty_spaces=[]
        values_of_empty_spaces=[]
        
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    empty_spaces.append([i,j])
                    values_of_empty_spaces.append(0)
        
        a =empty_spaces[random.choice(list(enumerate(values_of_empty_spaces)))[0]]
        
        self.state[a[0]][a[1]]=self.random_agent_value
        





        
def learn_from_random_agent():
    """

    Bu fonsiyonun amacı boş bir değer fonksiyonu yaratıp, bir çok oyun oynayarak değer fonksiyonunun
    içeriğini oynanan her adımda V (s) ← V (s) + α[V(s0)−V(S1)]    formülü ile güncellemektir.
    Öğrenmeyi rastgele hareketler yapan bir oyuncu üzerinden gerçekleştireceğiz.

    alpha ====> öğrenme katsayısı
    Öğrenme katsayısı yaptığımız her adım değer fonksiyonu güncellemesindeki [V(s0)−V(S1)] adımının katsayısı olacak ve
    her yeni deneyimin hafızamıza ne kadar etki bırakacağının katsayısı olacak.

    epsilon  ====>  rastgele seçim katsayısı
    Kullanacak olduğumuz epsilon-greedy politikasını bizi bulduğumuz ilk maksimum değere takılı kalmayıp ortamın ödüllerini
    epsilon olasılıkla keşif etmemizi sağlayacak.

    value_table  ====> ortamdaki her durumun için; içinde bulunması ne kadar güzel bir durum olduğunu anlatan,
    ortamın durum sayısı kadar skaler değer barındıran matris.  ortamdaki durum sayısı 9 ise ===> len(value_func)=9

    episodes   =====>öğrenme aşamasında kaç oyun oynatacağız.
    
    This function is for learning the value function with playing random agent
    Random agent just play randomly to valid states
    formula of value function estimation is ======>    V (s) ← V (s) + α[V(s0)−V(S1)]



    action -----> [i,j]  şeklinde gösterildiği için value function hesabında action(state) kullandım.
    """

    episodes=100
    value_table =[0,0,0,0,0,0,0,0,0]
   
    for i in range(episodes):
        eps=0.1
        env=Env(random_play=1)
        total_reward=0
        
        done=0
        agent_value="X"   # it must be X for random play
        state=env.state
        old_a = [0,0]
        alfa=0.99
        while not done:
           

            #print(state)
            
            empty_spaces=[]
            values_of_empty_spaces=[]
            
            for i in range(3):
                for j in range(3):
                    if state[i][j] == 0:
                        empty_spaces.append([i,j])
                        values_of_empty_spaces.append(value_table[i*3+j])




                        
            
            if np.random.random() < eps or np.sum(value_table) == 0:
               
                
                a =empty_spaces[random.choice(list(enumerate(values_of_empty_spaces)))[0]]
            else:
                # select the action with largest q value in state s
                a = empty_spaces[np.argmax(values_of_empty_spaces)]
           
           
            new_s,reward,done,winner = env.step(a,agent_value)
            total_reward=reward+total_reward
            if (state!=[[0,0,0],[0,0,0],[0,0,0]]):
                print(a)
                value_table[old_a[0]*3+old_a[1]] =value_table[old_a[0]*3+old_a[1]] + alfa*(reward-value_table[old_a[0]*3+old_a[1]])
            else:
                value_table[a[0]*3+a[1]] =alfa*reward
            old_a=a
            state=new_s
            
            if(done==1):
                print("Winner is  :"+ str(winner))
                print(value_table)

                
    


learn_from_random_agent()


