import json

cluster_x_size = 2
cluster_y_size = 4
buffer_num = 4
pattern_len = 8


def put_balls(balls, bowls):
    def generate(balls, bowls, current):
        if bowls == 1:
            res.append(current + [balls])
            return
        for i in range(balls + 1):
            generate(balls - i, bowls - 1, current + [i])
    
    res = []
    generate(balls, bowls, [])
    return res

def cost_model():
    return 0

def search():
    noc_group_size = 1
    schedule_id = 0
    schedule_list = []
    while noc_group_size <= max(cluster_x_size, cluster_y_size):
        # print("noc_group_size={}".format(noc_group_size))
        
        # schedule_per_blk = [
        #             {"tileid": 0, "src": "gmem"},
        #             {"tileid": 1, "src": "blk(0,0)"},
        #             {"tileid": 2, "src": "blk(0,0)"},
        #             {"tileid": 3, "src": "blk(0,0)"}
        #         ]
        
        noc_group_pair_num = 1
        while noc_group_pair_num * noc_group_size <= pattern_len:
            # print(" noc_group_pair_num={}".format(noc_group_pair_num))
            k_list = []
            for outer in range(noc_group_pair_num):
                for inner in range(noc_group_size - 1):
                    k_list.append(1 + outer * noc_group_size + inner)
            print("k_list,", k_list)

            indep_tile_num = pattern_len - noc_group_pair_num * noc_group_size
            bowl_num = noc_group_pair_num * noc_group_size + 1
            insert_indep_tile_list = put_balls(indep_tile_num, bowl_num)
            print("insert_indep_tile_list,", insert_indep_tile_list)
            filtered = [i for i in insert_indep_tile_list if all(i[k] < buffer_num - 2 for k in k_list)]
            # print(filtered)
            for f in filtered:
                schedule =  {
                                "schedule_id": schedule_id, 
                                "noc_group_size": noc_group_size, 
                                "noc_group_pair_num": noc_group_pair_num, 
                                "schedule": []
                            }
                base_order = []
                noc_group_tile_id = 0
                indep_tile_id = noc_group_pair_num * noc_group_size
                for i in f:
                    for j in range(i):
                        base_order.append(indep_tile_id)
                        indep_tile_id += 1
                    if noc_group_tile_id < noc_group_pair_num * noc_group_size:
                        base_order.append(noc_group_tile_id)
                        noc_group_tile_id += 1
                print("base_order:", base_order)
                
                
                for xid in range(cluster_x_size):
                    schedule_per_x = []
                    for yid in range(cluster_y_size):
                        schedule_per_blk = []
                        for iteration in range(pattern_len):
                            tile_id = base_order[iteration]
                            srcA = "gmem"
                            srcB = "gmem"
                            # tile in a NoC group
                            if tile_id < noc_group_pair_num * noc_group_size:
                                # if not the first iteration (first iteration need to load data from gmem)
                                if tile_id % noc_group_size != 0:
                                    src_A_yid = (yid + 1) % noc_group_size
                                    src_B_xid = (xid + 1) % noc_group_size
                                    if (src_A_yid < cluster_y_size):
                                        srcA = "blk(" + str(xid) + "," + str(src_A_yid + (yid - yid % noc_group_size)) + ")"
                                    else:
                                        srcA = "gmem"
                                    if (src_B_xid < cluster_x_size):
                                        srcB = "blk(" + str(src_B_xid + (xid - xid % noc_group_size)) + "," + str(yid) + ")"
                                    else:
                                        srcB = "gmem"
                                tile_id = ((tile_id + xid + yid) % noc_group_size) + (tile_id - tile_id % noc_group_size)
                            schedule_per_blk.append({"tileid": tile_id, "srcA": srcA, "srcB": srcB})
                        schedule_per_x.append({
                                                
                                                "blockIdx.x": xid, 
                                                "blockIdx.y": yid, 
                                                "schedule_per_blk": schedule_per_blk
                                            })
                    schedule["schedule"].append(schedule_per_x)
                schedule_list.append(schedule)
                schedule_id += 1
                if (noc_group_size == 1):
                    break
            noc_group_pair_num += 1
            if (noc_group_size == 1):
                break
        noc_group_size *= 2
    return schedule_list


if __name__ == '__main__':
    print("searching...")
    # schedule_list = [
    #     [
    #         [{"blockIdx.x": 0, "blockIdx.y": 0, "tileid": [0,1,2,3,4,5,6,7]}, {"blockIdx.x": 0, "blockIdx.y": 1, "tileid": [0,1,2,3,4,5,6,7]}],
    #         [{"blockIdx.x": 1, "blockIdx.y": 0, "tileid": [0,1,2,3,4,5,6,7]}, {"blockIdx.x": 1, "blockIdx.y": 1, "tileid": [0,1,2,3,4,5,6,7]}]
    #     ],
    #     [
    #         [{"blockIdx.x": 0, "blockIdx.y": 0, "tileid": [0,1,2,3,4,5,6,7]}, {"blockIdx.x": 0, "blockIdx.y": 1, "tileid": [1,0,2,3,4,5,6,7]}],
    #         [{"blockIdx.x": 1, "blockIdx.y": 0, "tileid": [0,1,2,3,4,5,6,7]}, {"blockIdx.x": 1, "blockIdx.y": 1, "tileid": [1,0,2,3,4,5,6,7]}]
    #     ],
    # ]

    schedule_list = search()
    print("find {} schedules".format(len(schedule_list)))
    # with open('schedule.json', 'w') as file:
    #     json.dump(schedule_list, file, indent=4)