import json
from test import *
import os

channel_list = ['express', 'hum', 'ptv']
channel_list_2 = ['test']

#*********************************************************#
#                       Main
#*********************************************************#
if __name__ == '__main__':
    """
    print(os.getcwd())
    lol = parse_anno_file("./Manual_Anno/Hum/hum_tv/annotations.xml", "2499.jpg")
    xtl = float(lol[0]['shapes'][0]['xtl'])
    print("xtl - 100: ", xtl - 100)
    print("XTL: ", lol[0]['shapes'][0]['xtl'])
    """

    for channel_name in channel_list:
        preds = os.path.join(channel_name, "infered_results")
        annotated = os.path.join(channel_name, "infered_results_final")
        print("**********************")
        print("Channel: ", channel_name)
        det_TP = 0
        det_FP = 0
        det_FN = 0
        rec_TP = 0
        rec_FP = 0
        rec_FN = 0
        for filename in os.listdir(preds):
            with open(os.path.join(preds, filename)) as f:
                pred_data = json.load(f)
            with open(os.path.join(annotated, filename)) as g:
                anot_data = json.load(g)
            #print("File: ", filename)
            #print("Prediction: ", len(pred_data['Label']))
            #print("Annotation: ", len(anot_data['Label']))
            missed = 0
            det_local_tp = 0
            rec_local_tp = 0
            
            unk_pred = 0
            unk_anot = 0
            for anot_counter_label in anot_data['Label']:
                if anot_counter_label == 'Unknown':
                    unk_anot += 1

            #print("Unk_Anot: ", unk_anot)

            for pred_counter_label in pred_data['Label']:
                if pred_counter_label == 'Unknown':
                    unk_pred += 1
            
            unk_counter = 0

            for i in range(len(pred_data['Bbox'])):
                validated = 0
                pred_box = pred_data['Bbox'][i]
                for anot_box in anot_data['Bbox']:
                    IOU = bb_intersection_over_union(pred_box, anot_box)
                    #print("IOU: ", IOU)
                    if IOU > 0.35:
                        validated = 1
                        #print("Truly Positive")
                if validated == 1:
                    det_TP += 1
                    det_local_tp += 1
                    
                    pred_label = pred_data['Label'][i]
                    #print("Recognition Validated for Label: ", pred_label)
                    rec_val = 0

                    if pred_label != 'small face' and pred_label != 'Invalid Pose':
                        for anot_label in anot_data['Label']:
                            if pred_label == anot_label:
                                #print("Pred Label: ", pred_label)
                                #print("Anot Label: ", anot_label)
                                if pred_label == 'Unknown':
                                    if unk_counter < unk_anot:
                                        #print("Validated")
                                        unk_counter += 1
                                        rec_val = 1
                                        break
                                    else:
                                        #print("Unknowns exceeded")
                                        rec_val = 0
                                    #    rec_FP += 1
                                        break

                                else:
                                    #print("Validated")
                                    rec_val = 1
                                    break
                    
                        if rec_val == 1:
                            rec_local_tp += 1
                            rec_TP += 1
                        else:
                            rec_FP += 1
            #print("Unk_Count: ", unk_counter)
            len_percieved_anot = 0
            for anot_counter_label in anot_data['Label']:
                if anot_counter_label != 'small face' and anot_counter_label != 'Invalid Pose':
                    len_percieved_anot += 1
            #print("Length Perceived: ", len_percieved_anot)
            rec_FN = rec_FN + len_percieved_anot - rec_local_tp

                            
            #print("Pred Boxes Length", len(pred_data['Bbox']))
            #print("Pred Boxes: ", pred_data['Label'])
            #print("Anot Boxes Length", len(anot_data['Bbox']))
            #print("Anot Boxes: ", anot_data['Label'])

            det_FP = det_FP + len(pred_data['Bbox']) - det_local_tp
            det_FN = det_FN + len(anot_data['Bbox']) - det_local_tp
        
        print("\n*************************")
        print("Detection")
        print("True Positives: ", det_TP)
        print("False Positives: ", det_FP)
        print("False Negatives: ", det_FN)

        det_precision = det_TP / (det_TP + det_FP)
        det_recall = det_TP / (det_TP + det_FN)
        det_f1s = 2 * ((det_precision * det_recall) / (det_precision + det_recall))

        print("Precision: ", det_precision)
        print("Recall: ", det_recall)
        print("F1 Score: ", det_f1s)

        print("\n*************************")
        print("Recognition")
        print("True Positives: ", rec_TP)
        print("False Positives: ", rec_FP)
        print("False Negatives: ", rec_FN)

        rec_precision = rec_TP / (rec_TP + rec_FP)
        rec_recall = rec_TP / (rec_TP + rec_FN)
        rec_f1s = 2 * ((rec_precision * rec_recall) / (rec_precision + rec_recall))

        print("Precision: ", rec_precision)
        print("Recall: ", rec_recall)
        print("F1 Score: ", rec_f1s)

        
        det_dict = {"True Positives" : det_TP, "False Positives" : det_FP, "False Negatives" : det_FN, "Precision" : det_precision, "Recall" : det_recall, "F1 Score" : det_f1s}
        rec_dict = {"True Positives" : rec_TP, "False Positives" : rec_FP, "False Negatives" : rec_FN, "Precision" : rec_precision, "Recall" : rec_recall, "F1 Score" : rec_f1s}
        
        dict = {"Channel" : channel_name, "Detection Metrics" : det_dict, "Recognition Metrics" : rec_dict}
        #print(dict)

        saver_string = channel_name + ".json"
        with open(saver_string, 'w') as f:
            json.dump(dict, f)

                    
                
        


    
    """
    #input_file = "./Manual_Anno/Hum/hum_tv_2/frames"
    ground_truth = "./Manual_Anno/Hum/hum_tv_2/infered_results"
    xml_file="./Manual_Anno/Hum/hum_tv_2/annotations.xml"
    input_manual = "./Manual_Anno/Hum/hum_tv_2/images"
    #match_annotations(ground_truth , xml_file , input_manual)
    genrate_final_annotation(ground_truth, xml_file, input_manual)
    #test_images(input_file , ground_truth , xml_file , input_manual)
    """