def WAND_Algo(query_terms, top_k, inverted_index):
    full_evaluation_count = 0

    # record the candidate list
    # each element is {term, Ub, index_values}
    candidate_list = []
    for t in range(0, len(query_terms)):
        term = query_terms[t]
        index_values = inverted_index[term]
        Ub = max(map(lambda x: x[1], index_values))
        candidate_list.append([term, Ub, index_values])

    threshold = -float('Inf')
    Ans = []
    while len(candidate_list) != 0:
        def get_Doc_id(candidate):
            return candidate[2][0][0]
        # sorted by DocID
        candidate_list.sort(key=get_Doc_id)
        score_limit = 0
        pivot = 0
        # find pivot
        while pivot < len(candidate_list):
            tmp_score_limit = score_limit + candidate_list[pivot][1]
            if tmp_score_limit > threshold:
                break
            score_limit = tmp_score_limit
            pivot = pivot + 1
        # if the sum of all Ub cannot reach threshold, end the function
        if pivot == len(candidate_list):
            break
        # calculate the target Doc
        if candidate_list[0][2][0][0] == candidate_list[pivot][2][0][0]:
            full_evaluation_count += 1
            # calculate score
            s = 0
            t = 0
            Doc_id = candidate_list[pivot][2][0][0]
            next_posting_list = []
            while t < len(candidate_list) and candidate_list[t][2][0][0] == candidate_list[pivot][2][0][0]:
                s = s + candidate_list[t][2][0][1]
                next_posting_list.append(candidate_list[t])
                t = t + 1
            # start next posting for each element
            for candidate in next_posting_list:
                if len(candidate[2]) == 1:
                    candidate_list.remove(candidate)
                else:
                    candidate[2] = candidate[2][1:]
            # add answer
            if s > threshold:
                Ans.append((s, Doc_id))
                if len(Ans) > top_k:
                    # find the element with smallest score
                    smallest = Ans[0]
                    for answer in Ans:
                        if smallest[0] > answer[0]:
                            smallest = answer
                        # if 2 answers has same score, remove whose doc id is bigger
                        elif smallest[0] == answer[0] and smallest[1] < answer[1]:
                            smallest = answer
                    Ans.remove(smallest)
                    # update threshold
                    threshold = min(map(lambda x: x[0], Ans))
        else:
            remove_list = []
            for t in range(pivot):
                # do skip-to for each element
                while candidate_list[t][2][0][0] < candidate_list[pivot][2][0][0]:
                    if len(candidate_list[t][2]) == 1:
                        remove_list.append(candidate_list[t])
                        break
                    else:
                        candidate_list[t][2] = candidate_list[t][2][1:]
            for candidate in remove_list:
                candidate_list.remove(candidate)

    # sort Ans by -score and Doc id
    def Ans_key(element):
        return -element[0], element[1]
    Ans.sort(key=Ans_key)

    return Ans, full_evaluation_count

