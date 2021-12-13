import numpy as np

"""
columnCombine: Combine multiple sparse columns of the matrix into a group,
each group has high density without serious value conflicts between columns

Parameters:
    matrix: the sparse matrix to process
    alpha: threshold of maximum #column to combine in a group
    gamma: threshold of maximum #conflict per row in a group.
    #conflicts = max(#nonzero - 1, 0)

Return value:
    groups: A list of column group
    Each group contains index of matrix columns packed in this group
    e.g. groups = [[1, 2, 4], [3]]
    The 1st, 2nd, 4th columns are combined together, while the 3rd column is seperate.
"""
def columnCombine(matrix, alpha=3, gamma=2):
    columns = np.hsplit(matrix, matrix.shape[1])
    
    groups = []
    
    for idx, col in enumerate(columns):
        best_group = None
        best_density = 0
        
        for group in groups:
            # Current column cannot be added
            # if the group reaches column capacity alpha
            if len(group) == alpha:
                continue
                
            # Construct a temproary matrix
            # with the column group
            # and current column as columns
            arrays = [columns[idx] for idx in group]
            arrays.append(col)
            matrix = np.hstack(arrays)
 
            nonzero_per_row = np.count_nonzero(matrix, axis=1)

            # Current column cannot be added
            # if the row with max(#conflict) reaches gamma
            if np.max(nonzero_per_row) - 1 > gamma:
                continue
                
            # Find the group with highest density
            # after combined with current column
            density = np.count_nonzero(nonzero_per_row)
            
            if best_density < density:
                best_density = density
                best_group = group

        # If no group can add current column
        # create a new group for it
        if not best_group:
            groups.append([idx])
        # else add the column to the group
        # with highest density after combination 
        else:
            best_group.append(idx)
    #print("groups are:")
    #for group in groups:
    #    print(np.hstack([columns[idx] for idx in group]))
    return groups


"""
structuredPruneMask: Get the mask of parameters pruned for eliminating column row conflicts 

Parameters:
    matrix: Unstructured pruned sparse matrix
    Lowered from conv2d weight filters
    e.g. matrix = array([[1, 0, 2, 0, 3],
                         [1, 1, 0, 0, 1],
                         [1, 2, 0, 1, 0],
                         [0, 0, 0, 1, 0]])
    [1, 0, 2, 0, 3] is the first row of matrix

    groups: A list of column group
    Each group contains index of matrix columns packed in this group
    e.g. groups = [[1, 2, 4], [3]]
    The 1st, 2nd, 4th columns are combined together, while the 3rd column is seperate.

Return value:
    A filter mask showing which values should be pruned after column combination
"""
def structuredPruneMask(matrix, groups):
    # Initialize mask to all-zero
    mask = np.zeros_like(matrix)
    for group in groups:
        # Slice columns in the group to get submatrix
        submatrix = matrix[:, group]
        
        # Get the max abs value's column index for every row
        submatrix_max_val_col_idx = np.argmax(abs(submatrix), axis=1)
        
        # Map submatrix column index -> original matrix column index
        max_val_col_idx = [group[i] for i in submatrix_max_val_col_idx]
        
        # Set these indexes, which correspond to unpruned values
        # in original matrix to "True"
        mask[np.arange(mask.shape[0]), max_val_col_idx] = True
    
    return mask


"""
example usage of column combine APIs

matrix = np.array([[1, 0, 2, 0, 3],
                   [1, 1, 0, 0, 1],
                   [1, 2, 0, 1, 0],
                   [0, 0, 0, 1, 0]])

groups = columnCombine(matrix.T, 3, 1)

mask = structuredPruneMask(matrix.T, groups)
"""
