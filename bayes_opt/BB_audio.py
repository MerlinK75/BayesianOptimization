import numpy as np
import pygame
import sounddevice
import random
import json


###The yputube videos had a range from 126 to 135 for alpha and 100 and 129 for beta
###callback function, that calculates the start and stop angle of each wave to make a continouos wave to remove harmonics

class Audio:
    def __init__(self, 
                # parent_current, parent_desired, Cur_Queue, Des_Queue
                 ):
        self.freq = 166
        self.cons_freq = 126
        self.sample_rate = 48_000
        self.increment = 0.1
        self.phase = 0
        self.phaseC = 0
        self.dummy_BB = 0
        self.dummy_DATA = 0
        self.thresh = 0


        # self.parent_current = parent_current
        # self.parent_desired = parent_desired
        # self.Cur_Queue = Cur_Queue
        # self.Des_Queue = Des_Queue

        pygame.init()
        pygame.display.set_mode(size=(320, 240))
        pygame.display.set_caption('Playing Binaural Beats')

    def audio_callback(self, 
        outdata: np.ndarray, frames: int, time: 'CData', status: sounddevice.CallbackFlags,
    ) -> None:
        omega = 2*np.pi*self.freq/self.sample_rate #Angular frequency
        start_angle = self.phase
        stop_angle = self.phase + frames*omega
        #constant_angle = phase + frames*2*np.pi*cons_freq/sample_rate
        self.phase = np.fmod(stop_angle, 2*np.pi)

        arg = np.linspace(
            start=start_angle,
            stop=stop_angle,
            num=frames,
        ) #Linear interpolation between the start and stop angle which can be input into the sine wave below
        omegaC = 2*np.pi*self.cons_freq/self.sample_rate #Angular frequency
        start_angleC = self.phaseC
        stop_angleC = self.phaseC + frames*omegaC
        self.phaseC = np.fmod(stop_angleC, 2*np.pi)

        argC = np.linspace(
            start=start_angleC,
            stop=stop_angleC,
            num=frames,
        ) 

        outdata[:, 1] = 10_000*np.sin(arg)
        outdata[:, 0] = 10_000*np.sin(argC)

    #If the other frequency is far use this to move the frequency
    def step(self, direction: int) -> None:
        self.freq = max(1, min(self.sample_rate, self.freq + direction*self.increment)) 
        print(f'Frequency: {self.freq-self.cons_freq:.1f} Hz')
        ##Deletes previous data
        # if self.dummy_BB == 0:
        #     with open('BBSham.json', 'w') as f:
        #         self.dummy_BB = 1
        #         pass
        # ###Appends new data and the time stamp they occur
        # with open('BBSham.json','a') as f:
        #     f.write(json.dumps({'BB': np.round(self.freq-self.cons_freq, 1)}))
        #     f.write("\u000a")
        #     f.write(json.dumps({'time (s)': pygame.time.get_ticks()/1000}))
        #     f.write("\u000a")
    def freq_assign(self, new_freq) -> None:
        self.freq = self.cons_freq + new_freq
        print(f"Assigned: {new_freq}Hz")
        ###Add steps and wait times in this function

    def step_init(self) -> None:
        ##The calibration so we can get an initial mapping of BB and alpha frequency
        x = np.linspace(40, 10, 31) ###This range depends on literature first to see the effect each band has on attention and relaxation
        pygame.time.delay(5000)
        for value in x:
            while True:
                if self.freq-self.cons_freq > value:
                    self.step(-1)
                    pygame.time.delay(200)
                else: 
                    #self.recorder()
                    self.thresh = np.round(pygame.time.get_ticks()/1000, 1)
                    
                    break 

    # def recorder(self) -> None:
    #         if self.dummy_DATA == 0:
    #             print("Delete past data")
    #             with open('Data_Sham.json', 'w') as f:
    #                 self.dummy_DATA = 1
    #                 pass
    #             #This is the changed w after the update function has been activated and previous error
    #         with open('Data_Sham.json','a') as f:
    #             self.Cur_Queue.put(True)
    #             self.Des_Queue.put(True)
    #             print("Start Cur_Queue")
    #             m_alpha = np.round(self.parent_current.recv(), 1)
    #             mean_alpha = np.mean(m_alpha)
    #             des_alpha = np.round(self.parent_desired.recv(), 1)

    #             self.Cur_Queue.put(False)
    #             self.Des_Queue.put(False)

    #             error = [x - mean_alpha for x in des_alpha]
    #             f.write(json.dumps({'alpha_error': (min(error, key=abs)).tolist()})) #Print the difference to the closest value of mean and desired alpha
    #             f.write("\u000a")
    #             f.write(json.dumps({'Desired_alpha_range': des_alpha.tolist()})) #Print the difference to the closest value of mean and desired alpha
    #             f.write("\u000a")
    #             f.write(json.dumps({'Mean_alpha': mean_alpha.tolist()})) #Print the difference to the closest value of mean and desired alpha
    #             f.write("\u000a")
    #             f.write(json.dumps({'time(s)': pygame.time.get_ticks()/1000}))
    #             f.write("\u000a")


    def main(self) -> None:
        #Initiate pygame window

        with sounddevice.OutputStream(
            callback=self.audio_callback, channels=2, samplerate=self.sample_rate, dtype='int16',
        ) as stream:
            stream.start()
            #self.step_init()
            # while True:

            #     # if np.round(pygame.time.get_ticks()/1000, 1)- self.thresh > 5:
            #     #     self.recorder()
            #     #     self.thresh = np.round(pygame.time.get_ticks()/1000, 1)

                
            #     pygame.time.delay(200)
            #     x = random.choice([-1, 1])
            #     if x == -1:
            #         if self.freq-self.cons_freq <=8:
            #             self.step(1)
            #         else:
            #             self.step(-1)
            #     else:
            #         if self.freq-self.cons_freq >=14:
            #             self.step(-1)
            #         else:
            #             self.step(1)

                # for e in pygame.event.get():
                #     if e.type == pygame.QUIT:
                #         return
                # pressed = pygame.key.get_pressed()
                # if pressed[pygame.K_DOWN]:
                #     self.step(-1)
                # elif pressed[pygame.K_UP]:
                #     self.step(1)


# if __name__ == '__main__':
#     sham_BB = sham()
#     sham_BB.main()
