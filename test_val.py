import shutil
import random
import os

train_dir_str='/home/dany/Documents/CASIA_train_232'

#test_dir_str='/home/deep-visage/Documents/test_299'

val_dir_str='/home/dany/Documents/CASIA_val_232'

#shutil.move('/home/deep-visage/Documents/train_299/n000904/0210_01_0.png', '/home/deep-visage/Desktop/n000904/0210_01_0.png')


image_count=0
for fold in os.listdir(train_dir_str):

    sub_fold=os.path.join(train_dir_str, fold)
    sub_fold_list=os.listdir(sub_fold)

    other_sub_fold=os.path.join(val_dir_str, fold)
    #other_sub_fold_list = os.listdir(other_sub_fold)

    # other_sub_fold = os.path.join(test_dir_str, fold)
    #os.makedirs(other_sub_fold)

    print(image_count)

    if len(sub_fold_list) > 500:

        index=random.sample(range(1,499),10)

        for i in index:

            shutil.move(os.path.join(sub_fold, sub_fold_list[i]), os.path.join(other_sub_fold, sub_fold_list[i]))
            #print(os.path.join(sub_fold, sub_fold_list[i]))
            #print(os.path.join(other_sub_fold, sub_fold_list[i]))
        image_count += 10



    else:

        if len(sub_fold_list) > 200:

            index = random.sample(range(1, 199), 6)
            for i in index:
                shutil.move(os.path.join(sub_fold, sub_fold_list[i]), os.path.join(other_sub_fold, sub_fold_list[i]))
                #print(os.path.join(sub_fold, sub_fold_list[i]))
                #print(os.path.join(other_sub_fold, sub_fold_list[i]))
            image_count += 6

        else:

            if len(sub_fold_list) > 100:
                index = random.sample(range(1, 99), 4)
                for i in index:
                    shutil.move(os.path.join(sub_fold, sub_fold_list[i]),os.path.join(other_sub_fold, sub_fold_list[i]))
                    #print(os.path.join(sub_fold, sub_fold_list[i]))
                    #print(os.path.join(other_sub_fold, sub_fold_list[i]))
                image_count += 4

            else:

                if len(sub_fold_list) > 30:
                    index = random.sample(range(1, 29), 3)
                    for i in index:
                        shutil.move(os.path.join(sub_fold, sub_fold_list[i]),os.path.join(other_sub_fold, sub_fold_list[i]))
                        #print(os.path.join(sub_fold, sub_fold_list[i]))
                        #print(os.path.join(other_sub_fold, sub_fold_list[i]))
                    image_count += 3

                else:
                    pass
                    



                        














print(image_count)
print('ok')



