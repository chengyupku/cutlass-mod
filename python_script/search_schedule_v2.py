import json
import itertools

cluster_x_size = 4
cluster_y_size = 4
# require cluster_x_size >= 2, cluster_y_size >= 2
# assert(cluster_x_size, "cluster_x_size must >= 2")
# assert(cluster_y_size, "cluster_y_size must >= 2")
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

def solve(ans, pos, len, solutions):
    if (len <= 0):
        solutions.append(ans)
        return
    # no noc on the first iteration
    # solve(ans, pos + 1, len - 1)
    # print("!", ans, pos, len)
    size = 2
    ans_no_put = ans.copy()
    solve(ans_no_put, pos + size, 0, solutions)
    while size <= len:
        ans_put = ans.copy()
        ans_put.append({"pos":pos, "size":size})
        solve(ans_put, pos + size, len - size, solutions)
        size *= 2

def select_schedule(in_list, check_list):
    if not all(in_list[index] <= buffer_num - 2 for index in check_list):
        return False
    if (not all(in_list[index] > 0 for index in check_list)) and (sum(in_list) - sum(in_list[index] > 0 for index in check_list) > 0):
        return False
    return True


def find_schedule(total_len, solutions):
    ans_num = 0
    filtered_list = []
    for solution in solutions:
        noc_group_tile_num = sum([noc_group['size'] for noc_group in solution])
        indep_tile_num = total_len - noc_group_tile_num
        # print(indep_tile_num)
        insert_indep_tiles = put_balls(indep_tile_num, noc_group_tile_num + 1)
        # print('before:', len(insert_indep_tiles))
        check_index = []
        for noc_group in solution:
            for index in range(noc_group['pos'] + 1, noc_group['pos'] + noc_group['size']):
                check_index.append(index)
        # print('check_index:', check_index)
        filtered = [i for i in insert_indep_tiles if select_schedule(i, check_index)]
        filtered_list.append(filtered)
        # print('filtered:', filtered)
        # print('after:', len(filtered))
        ans_num += len(filtered)
    # print("ans_num", ans_num)
    return filtered_list

def replace_elements(original_list, index_list, new_elements):
    result_list = original_list.copy()
    sorted_indices = sorted(index_list, reverse=True)
    for index in sorted_indices:
        if index < len(result_list):
            result_list.pop(index)
            for element in reversed(new_elements):
                result_list.insert(index, element)
    # print(result_list)
    return result_list

def get_info_from_tile_id(noc_groups, tile_id):
    # 'noc_groups':[{'pos': 0, 'size': 4}, {'pos': 4, 'size': 2}]
    # 'indep_tile':[0, 0, 0, 1, 0, 1, 0]
    for noc_group in noc_groups:
        if tile_id < noc_group['pos'] + noc_group['size']:
            return noc_group['pos'], noc_group['size']
    raise RuntimeError('get_info_from_tile_id error!')

if __name__ == "__main__":
    ans = []
    solutions = []
    sub_solutions_list = []
    sub_insert_indep_tiles_list = []
    solve(ans, 0, pattern_len, solutions)
    insert_indep_tile_list = find_schedule(pattern_len, solutions)

    plan_A_list = []
    plan_B_list = []
    # plan_A/B_list:[
    #     {
    #         'noc_groups':[{'pos': 0, 'size': 4}, {'pos': 4, 'size': 2}, {'pos': 6, 'size': 2}...]
    #         'indep_tile':[0, 0, 0, 1, 0, 0, 0, 0, 0]
    #     },
    #     {
    #         ...
    #     },
    #     ...
    # ]
    # solutions = [[{'pos': 0, 'size': 2}, {'pos': 2, 'size': 4}]]
    # insert_indep_tile_list = [[[0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    for solution, insert_indep_tile in zip(solutions, insert_indep_tile_list):
        print("solution:", solution)
        # print("insert_indep_tile:", insert_indep_tile)
        # print("---------------------------------------")
        if len(solution) > 0:
            noc_group_size = [noc_group['size'] for noc_group in solution]
            max_noc_group_size = max(noc_group_size)
            for iit in insert_indep_tile:
                if (max_noc_group_size > cluster_x_size and max_noc_group_size > cluster_y_size):
                    #  NoC group bigger than cluster size, drop it
                    continue
                elif (max_noc_group_size > cluster_y_size):
                    A_noc_groups = []
                    B_noc_groups = solution
                    for noc_group in solution:
                        if noc_group['size'] > cluster_y_size:
                            for k in range(int(noc_group['size'] / cluster_y_size)):
                                A_noc_groups.append({'pos':noc_group['pos'] + k * cluster_y_size, 'size':cluster_y_size})
                        else:
                            A_noc_groups.append(noc_group)

                    plan_A = {}
                    plan_B = {}
                    plan_A['noc_groups'] = A_noc_groups
                    plan_B['noc_groups'] = B_noc_groups
                    plan_A['indep_tile'] = iit
                    plan_B['indep_tile'] = iit
                    plan_A_list.append(plan_A)
                    plan_B_list.append(plan_B)
                elif (max_noc_group_size > cluster_x_size):
                    A_noc_groups = solution
                    B_noc_groups = []
                    for noc_group in solution:
                        if noc_group['size'] > cluster_x_size:
                            for k in range(int(noc_group['size'] / cluster_x_size)):
                                B_noc_groups.append({'pos':noc_group['pos'] + k * cluster_x_size, 'size':cluster_x_size})
                        else:
                            B_noc_groups.append(noc_group)
                    plan_A = {}
                    plan_B = {}
                    plan_A['noc_groups'] = A_noc_groups
                    plan_B['noc_groups'] = B_noc_groups
                    plan_A['indep_tile'] = iit
                    plan_B['indep_tile'] = iit
                    plan_A_list.append(plan_A)
                    plan_B_list.append(plan_B)

        for iit in insert_indep_tile:
            plan_A = {}
            plan_B = {}
            plan_A['noc_groups'] = solution
            plan_B['noc_groups'] = solution
            plan_A['indep_tile'] = iit
            plan_B['indep_tile'] = iit
            plan_A_list.append(plan_A)
            plan_B_list.append(plan_B)
    
    # for plan_A, plan_B in zip(plan_A_list, plan_B_list):
    #     print('plan_A:', plan_A)
    #     print('plan_B:', plan_B)
    #     print('-----------------------')


    schedule_id = 0
    schedule_list = []
    for plan_A, plan_B in zip(plan_A_list, plan_B_list):
        print('-------------------------')
        print('plan_A:', plan_A)
        print('plan_B:', plan_B)
        schedule =  {
                        "schedule_id": schedule_id, 
                        "schedule": []
                    }
        base_order = []
        assert(len(plan_A['indep_tile']) == len(plan_B['indep_tile']))
        for pa, pb in zip(plan_A['indep_tile'], plan_B['indep_tile']):
            assert(pa == pb)
        noc_tile_id = 0
        noc_tile_num = len(plan_A['indep_tile']) - 1
        noc_lead_pos_A = [g['pos'] for g in plan_A['noc_groups']]
        noc_lead_pos_B = [g['pos'] for g in plan_B['noc_groups']]
        indep_tile_id = noc_tile_num
        for it in range(noc_tile_num + 1):
            for _ in range(plan_A['indep_tile'][it]):
                base_order.append(indep_tile_id)
                indep_tile_id += 1
            if (it < noc_tile_num):
                base_order.append(noc_tile_id)
                noc_tile_id += 1
        # print(base_order)

        tile_id_all = []
        for xid in range(cluster_x_size):
            tile_id_x = []
            for yid in range(cluster_y_size):
                tile_id_y = []
                for iteration in range(pattern_len):
                    tile_id = base_order[iteration]
                    # tile in a NoC group
                    if tile_id < noc_tile_num:
                        group_base_id_A, noc_group_size_A = get_info_from_tile_id(plan_A['noc_groups'], tile_id)
                        group_base_id_B, noc_group_size_B = get_info_from_tile_id(plan_B['noc_groups'], tile_id)
                        if noc_group_size_A >= noc_group_size_B:
                            tile_id = ((yid + ((tile_id + xid - group_base_id_B) % noc_group_size_B) + group_base_id_B - group_base_id_A) % noc_group_size_A) + group_base_id_A
                        else:
                            tile_id = ((xid + ((tile_id + yid - group_base_id_A) % noc_group_size_A) + group_base_id_A - group_base_id_B) % noc_group_size_B) + group_base_id_B
                    tile_id_y.append(tile_id)
                tile_id_x.append(tile_id_y)
            tile_id_all.append(tile_id_x)
        print(schedule_id)
        print(tile_id_all)


        for xid in range(cluster_x_size):
            schedule_per_x = []
            for yid in range(cluster_y_size):
                schedule_per_blk = []
                last_noc_iter_pos = 1e4
                for iteration in range(pattern_len):
                    base_tile_id = base_order[iteration]
                    srcA = "gmem"
                    srcB = "gmem"

                    # tile in a NoC group
                    if base_tile_id < noc_tile_num:
                        group_base_id_A, noc_group_size_A = get_info_from_tile_id(plan_A['noc_groups'], base_tile_id)
                        group_base_id_B, noc_group_size_B = get_info_from_tile_id(plan_B['noc_groups'], base_tile_id)
                        # if not the first iteration (first iteration need to load data from gmem)
                        if base_tile_id not in noc_lead_pos_A:
                            src_A_yid = -1
                            for src_y in range(int((yid / noc_group_size_A)) * noc_group_size_A, int((yid / noc_group_size_A + 1)) * noc_group_size_A):
                                if src_y >= cluster_y_size:
                                    continue
                                # assert(last_noc_iter_pos >= 0, "last_noc_iter_pos error!")
                                if tile_id_all[xid][src_y][last_noc_iter_pos] == tile_id_all[xid][yid][iteration]:
                                    src_A_yid = src_y
                            if (src_A_yid != -1):
                                srcA = "blk(" + str(xid) + "," + str(src_A_yid) + ")"
                            else:
                                srcA = "gmem"
                        if base_tile_id not in noc_lead_pos_B:
                            src_B_xid = -1
                            for src_x in range(int((xid / noc_group_size_B)) * noc_group_size_B, int((xid / noc_group_size_B + 1)) * noc_group_size_B):
                                if src_x >= cluster_x_size:
                                    continue
                                # assert(last_noc_iter_pos >= 0, "last_noc_iter_pos error!")
                                if tile_id_all[src_x][yid][last_noc_iter_pos] == tile_id_all[xid][yid][iteration]:
                                    src_B_xid = src_x
                            if (src_B_xid != -1):
                                srcB = "blk(" + str(src_B_xid) + "," + str(yid) + ")"
                            else:
                                srcB = "gmem"
                        last_noc_iter_pos = iteration
                    schedule_per_blk.append({"tileid": tile_id_all[xid][yid][iteration], "srcA": srcA, "srcB": srcB})
                schedule_per_x.append({
                                        
                                        "blockIdx.x": xid, 
                                        "blockIdx.y": yid, 
                                        "schedule_per_blk": schedule_per_blk
                                    })
            schedule["schedule"].append(schedule_per_x)
        schedule_list.append(schedule)
        schedule_id += 1
    # print(schedule_list)
    with open('schedule_4x4x1.json', 'w') as file:
        json.dump(schedule_list, file, indent=4)
        