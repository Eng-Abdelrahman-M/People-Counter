
import torch
import cv2
import os


class Model:


    def __init__(self):
        self.model = torch.hub.load('core/yolov5', 'custom',path='core/yolov5/models/crowdhuman_yolov5m.pt',source='local')

    #function to calculate the intersection area between two rectangles. 
    @staticmethod
    def area(a, b):  
        dx = min(a['x2'], b['x2']) - max(a['x1'], b['x1'])
        dy = min(a['y2'], b['y2']) - max(a['y1'], b['y1'])

        a1= (a['x2']-a['x1']) * (a['y2']-a['y1'])
        a2= (b['x2']-b['x1']) * (b['y2']-b['y1'])

        if (dx>=0) and (dy>=0):
            return (dx*dy)/ min(a1,a2)
        else:
            return 0

    def detect(self,path,ROIs,out_path):
        cap = cv2.VideoCapture(path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #iterate over video frames or image sequence
        while(True):
            ret, frame_o = cap.read()
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if ret==True:

                frame = cv2.cvtColor(frame_o.copy(), cv2.COLOR_BGR2RGB)

                # Inference
                self.model.classes =0
                results = self.model(frame)
                results_bound = results.xyxy[0][:,:4]

                #initiate the counters
                counts = dict.fromkeys(ROIs.keys(), 0)
                # Results
                for rr in results_bound:
                    rr= rr.tolist()
                    people_dictionary = dict(zip(['x1','y1','x2','y2'], rr))
                    #draw rectangle for every person
                    frame_o = cv2.rectangle(frame_o, (int(rr[0]),int(rr[1])), (int(rr[2]),int(rr[3])), (0,255,0), 2)
                    #check the number of persons in every region
                    for k in ROIs.keys():
                        if Model.area(ROIs[k],people_dictionary)>0.5:
                            counts[k]+=1

                #draw regions rectangles and counters.
                for key, value in ROIs.items():
                    frame_o = cv2.rectangle(frame_o, (value['x1'],value['y1']), (value['x2'],value['y2']), (255,0,0), 2)
                    if value['y1'] > 25:
                        frame_o = cv2.putText(frame_o, key+': '+str(counts[key]), (value['x1']+1,value['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    else:
                        frame_o = cv2.putText(frame_o, key+': '+str(counts[key]), (value['x1']+1,value['y1']+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
            
                print(os.path.join(out_path,"frame_{:04d}.jpg".format(int(pos_frame))))
                #save output images
                cv2.imwrite(os.path.join(out_path,"frame_{:04d}.jpg".format(int(pos_frame)-1)),frame_o)
            
            if pos_frame == total_frames :
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

        cap.release()