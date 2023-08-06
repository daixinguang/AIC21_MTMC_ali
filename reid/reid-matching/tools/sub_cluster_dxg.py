from utils.filter import *
from utils.visual_rr import visual_rerank, visual_rerank_dxg
from sklearn.cluster import AgglomerativeClustering
import sys
sys.path.append('../../../')
from config import cfg
import numpy as np

def _calc_bev_pos(src_point_M, warpMatrix):
        '''点的透视变换，矩阵优化版
        Args:
            src_point_M():
            warpMatrix():
        '''
        one_column = np.ones((len(src_point_M), 1))
        src_point_M = np.column_stack((src_point_M, one_column))
        tar_pt = np.matmul(src_point_M, warpMatrix.T)
        tar_pt[:, 0] = tar_pt[:, 0] / tar_pt[:, 2]
        tar_pt[:, 1] = tar_pt[:, 1] / tar_pt[:, 2]
        return tar_pt[:, 0:2]
    
def get_sim_matrix(_cfg,cid_tid_dict,cid_tids, use_st_filter=True):
    count = len(cid_tids)
    print('count: ', count)
    #! dxg add TSMatch distance matrix 
    # 轨迹的坐标信息是 cid_tid_dict_new[(2, 1)]['tracklet']，这里有保存bbox，先根据该 bbox 得到 中心点3/4的坐标。再将其转换为bev坐标。添加新的字段:bev_tracklets: [],长度也和framelist一致
    
    old_cam = 0
    for c_t in cid_tid_dict:
        cam, tid = c_t
        if cam != old_cam:
            if cam==5:
                continue
            cam_hom = np.load(f'../../../reid/reid-matching-dxg/trans_cfg/c00{cam}_trans_bev.npy')
            old_cam = cam
        
        cam_tracklets = []
        for fi in cid_tid_dict[c_t]['frame_list']:
            bbox = cid_tid_dict[c_t]['tracklet'][fi]['bbox']
            px = bbox[0] + (bbox[2]-bbox[0]) / 2
            py = bbox[0] + (bbox[2]-bbox[0]) / 4 * 3
            cam_tracklets.append([px, py])
        
        cam_tracklets = np.array(cam_tracklets)
        bev_tracklets = _calc_bev_pos(cam_tracklets, cam_hom)
        print('tmp')
        cid_tid_dict[c_t]['bev_tracklets'] = bev_tracklets
        # 这里利用spi.BSpline去将轨迹重塑
        
    #! dxg calc TSmatch distance matrix
    #! 不同的bve
    q_ts_arr = np.array([cid_tid_dict[cid_tids[i]]['bev_tracklets'] for i in range(count)])
    g_ts_arr = np.array([cid_tid_dict[cid_tids[i]]['bev_tracklets'] for i in range(count)])
    #! dxg calc apparance distance matrix
    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1) # 对当前轨迹的特征做归一化
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32) # 这里是全部
    st_mask = intracam_ignore(st_mask, cid_tids) # 如果轨迹属于同一个相机，将st_mask的对应位置置为0。这里传进来的都是相邻相机的轨迹，也就是1[cam1], 2[cam1,cam2]
    #### dxg use_st_filter####
    # dxg 这是一个时空过滤，只能用于test场景的时空过滤！！
    if use_st_filter:
        st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)

    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    # visual_sim_matrix = visual_rerank_dxg(q_arr, g_arr, q_ts_arr, g_ts_arr, cid_tids, _cfg)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    print(visual_sim_matrix)
    
    
            
        
    
    
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def get_sim_matrix_ori(_cfg,cid_tid_dict,cid_tids):
    count = len(cid_tids)
    print('count: ', count)

    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)
    st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)

    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    visual_sim_matrix = visual_sim_matrix.astype('float32')
    print(visual_sim_matrix)
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict,cid_tids) # dxg 第一次聚类，得到从41进入的轨迹，从46离开的轨迹
    sub_labels = dict()
    dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                0.7,0.5,0.5,0.5,0.5]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c]) # 这里没报错是因为没执行进来，只有41和46摄像头才会执行到这里
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # 2ed cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids) # dxg 第二次聚类，专门除了41和46相机的轨迹！
    sub_labels = dict()
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        #### dxg modified get_sim_matrix ####
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c], use_st_filter=False) # 这样就减少了很大的相似矩阵
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-0.1, affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    # 3rd cluster
    # cid_tid_dict_new = combin_feature(cid_tid_dict,sub_cluster)
    # sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new, cid_tids)
    # cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.2, affinity='precomputed',
    #                                          linkage='complete').fit_predict(1 - sim_matrix)
    # labels = get_match(cluster_labels)
    return labels

if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    # scene_name = ['S02'] # dxg 
    # scene_cluster = [[6,7,8,9]] # dxg 
    scene_cluster=[]
    c = []
    for cam in os.listdir(cfg.DET_SOURCE_DIR):
        #### dxg ####
        # if cam == 'c005': # dxg no_c005
        #     pass
        c.append(int(cam[1:]))
    scene_cluster.append(c) # dxg
    mode, scenes = cfg.DET_SOURCE_DIR.split(os.sep)[-3:-1]

    fea_dir = f'./exp/viz/{mode}/{scenes}/trajectory/' # dxg
    cid_tid_dict = dict()

    for pkl_path in os.listdir(fea_dir):
        if pkl_path.find('5'): # dxg no_c005
            pass
        cid = int(pkl_path.split('.')[0][-3:])
        with open(opj(fea_dir, pkl_path),'rb') as f:
            lines = pickle.load(f)
        for line in lines:
            tracklet = lines[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster[0]])
    clu = get_labels(cfg,cid_tid_dict,cid_tids,score_thr=cfg.SCORE_THR)
    print('all_clu:', len(clu))
    new_clu = list()
    for c_list in clu:
        if len(c_list) <= 1: continue
        cam_list = [cid_tids[c][0] for c in c_list]
        if len(cam_list)!=len(set(cam_list)): continue
        new_clu.append([cid_tids[c] for c in c_list])
    print('new_clu: ', len(new_clu))

    all_clu = new_clu

    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open('test_cluster.pkl', 'wb'))
