import numpy as np
import torch
import torch.utils.data as data

class implicit_CF_dataset(data.Dataset):
    def __init__(self,user_count,item_count,rating_mat,num_negative_samples,interactions,RRD_interesting_items=None):
        # super(implicit_CF_dataset, self).__init__()
        super().__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.interactions = interactions
        self.num_negative_samples = num_negative_samples
        self.RRD_interesting_items = RRD_interesting_items
        self.train_arr = None

        if RRD_interesting_items is not None:
           self.num_b_user = RRD_interesting_items.size(0)
        else:
              self.num_b_user = -1

    def __len__(self):
        return len(self.interactions)*self.num_negative_samples

    def __getitem__(self, idx):
        assert self.train_arr # make sure the negative sampling has been done
        return {'user': self.train_arr[idx][0],'pos_item': self.train_arr[idx][1],'neg_item': self.train_arr[idx][2]}

    def negative_sampling(self):
        self.train_arr = []
        # Sampling a little more to ensure that there are enough negative samples even if some items do not meet the conditions
        sample_list=np.random.choice(list(range(self.item_count)),size=10*len(self.interactions)*self.num_negative_samples)

        sample_index=0
        for user,u_dict in self.rating_mat.items():
            pos_items=list(u_dict.keys())
            # Avoid selecting positive samples or interesting items when generating negative samples
            if self.RDD_interesting_items is not None and user<self.num_b_user:
                ignore_items=list(set(pos_items+self.RRD_interesting_items[user].tolist()))
            else:
                ignore_items=pos_items

            for pos_item in pos_items:
                cnt_negative_samples=0
                while cnt_negative_samples<self.num_negative_samples:
                    neg_item=sample_list[sample_index]
                    sample_index+=1
                    if neg_item not in ignore_items:
                        self.train_arr.append((user,pos_item,neg_item))
                        cnt_negative_samples+=1

class implicit_CF_dataset_AE(data.Dataset):
    def __init__(self,user_count,item_count,rating_mat=None,is_user_side=True,R=None):
        # super(implicit_CF_dataset_AE, self).__init__()
        super().__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.is_user_side = is_user_side

        # ensures that at least one of rating_mat or R is not empty, otherwise there is no source of data to initialize the matrix R
        assert rating_mat is not None or R is not None

        if R is not None:
            self.R = R
        else:
            self.R=torch.zeros((user_count,item_count))
            for user in rating_mat:
                pos_items=list(rating_mat[user].keys())
                self.R[user][pos_items]=1

        if not is_user_side:
            self.R=self.R.T#[user][pos_items] -> [pos_items][user]

    def __len__(self):
        if self.is_user_side:
            return self.user_count
        else:
            return self.item_count

    def __getitem__(self, user):
        return {'user':user,'rating_vec':self.R[user]}

    def negative_sampling(self):
        pass

class RDD_dataset_simple(data.Dataset):
    def __init__(self,interesting_items,score_mat,num_uninteresting_items):
        super().__init__()

        self.interesting_items=interesting_items
        self.score_mat=score_mat
        self.num_uninteresting_items=num_uninteresting_items
        self.uninteresting_items=None

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def sampling_for_uninteresting_items(self):
        print(f"Sampling_for_uninteresting_items({self.num_uninteresting_items})...")
        #interesting_items in score_mat have been set to nan
        self.uninteresting_items = torch.multinomial(self.score_mat, self.num_uninteresting_items, replacement=True)

    def get_samples(self,batch_user):
        interesting_samples=torch.index_select(self.interesting_items,0,batch_user)
        uninteresting_samples=torch.index_select(self.uninteresting_items,0,batch_user)
        return interesting_samples,uninteresting_samples

class IR_RRD_dataset_simple(data.Dataset):
    def __init__(self,interesting_users, score_mat, num_uninteresting_users):
        super().__init__()

        self.interesting_users = interesting_users
        self.score_mat = score_mat
        self.num_uninteresting_users = num_uninteresting_users
        self.uninteresting_users=None

        def __len__(self):
            pass

        def __getitem__(self, idx):
            pass

        def sampling_for_uninteresting_users(self):
            print(f"Sampling_for_uninteresting_users({self.num_uninteresting_users})...")
            self.uninteresting_users = torch.multinomial(self.score_mat, self.num_uninteresting_users, replacement=True)

        def get_samples(self, batch_item):
            interesting_users=torch.index_select(self.interesting_users,0,batch_item)
            uninteresting_users=torch.index_select(self.uninteresting_users,0,batch_item)
            return interesting_users,uninteresting_users



