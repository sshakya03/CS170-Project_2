import math
import copy
import time

def main():
    print("Welcome to Sazen Shakya's Feature Selection Algorithm.")
    
    start_time = time.time() # Start timer
    
    file_name = input("Type in the name of the file to test: ")
    algorithm = input("Type the number of the algorithm you want to run.\n\t1) Forward Selection\n\t2) Backward Elimination\n")
    data = [list(map(float, line.split())) for line in open(file_name)] # Read in dataset into data as a 2D list
    
    print(f"\nThis dataset has {len(data[0])-1} features (not including the class attribute), with {len(data)} instances.")
    set_of_features = list(range(1, len(data[0])))
    accuracy = leave_one_out_cross_validation(data, set_of_features,0) # 0 because no feature to add 
    print(f"Running nearest neighbor with all {len(data[0])-1} features, using \"leaving-one-out\" evaluation, I get an accuracy of {accuracy:.1f}%")
    print("Beginning search.")
    
    if algorithm == '1':
        forward_selection(data) 
    else:
        backward_elimination(data)
        
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    
    # Calculate elapsed time in hours, minutes, seconds
    hours = (elapsed_time / 3600)
    minutes = elapsed_time / 60
    seconds = int(elapsed_time % 60)

    print(f"\nTotal execution time: {hours:.2f} hours or {minutes:.2f} minutes or {seconds} seconds.")

def leave_one_out_cross_validation(data, currentSet, feature_to_add):
    if not currentSet: # If current set empty
        return calculate_empty_set_accuracy(data)
    
    copy_data = copy.deepcopy(data) # Create deepcopy of dataset
    columns_to_keep = currentSet.copy()

    if feature_to_add > 0:
        columns_to_keep.append(feature_to_add) # Add new feature to keep that column's data

    # Set all other feature's data to 0s
    for row in copy_data:
        for col_index in range(1, len(row)):
            if col_index not in columns_to_keep:
                row[col_index] = 0

    number_correctly_classified = 0
    for i in range(len(copy_data)):
        object_to_classify = copy_data[i][1:]
        label_object_to_classify = copy_data[i][0]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None
        for k in range(len(copy_data)):
            if k != i:
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(object_to_classify, copy_data[k][1:]))) # Euclidean Distance Formula

                if distance < nearest_neighbor_distance: # Update nearest distance if distance found is smaller
                    nearest_neighbor_distance = distance
                    nearest_neighbor_label = copy_data[k][0]

        if label_object_to_classify == nearest_neighbor_label: # Increment count if correctly classified
            number_correctly_classified += 1
        
    accuracy = (number_correctly_classified / len(copy_data)) * 100 # Calculate accuracy
    return accuracy

def calculate_empty_set_accuracy(data): 
    count_class_1 = sum(1 for row in data if row[0] == 1)
    count_class_2 = sum(1 for row in data if row[0] == 2)
    
    # Determines what class has the higher count
    majority_class_count = max(count_class_1, count_class_2)
    
    # Calculates default rate
    accuracy = (majority_class_count / len(data)) * 100
    return accuracy

def forward_selection(data):
    current_set_of_features = []
    
    # Variables to hold which set of features are the best
    best_accuracy = 0
    best_set_of_features = []

    for i in range(1, len(data[0])):
        feature_to_add_on_level = None
        best_accuracy_on_level = 0
        for k in range(1, len(data[0])):
            if k not in current_set_of_features: # Make sure not comparing a feature already in the set
                features_to_test = [k] + current_set_of_features
                accuracy = leave_one_out_cross_validation(data,features_to_test,k) # Calls classifier to get accuracy
                print(f"\tUsing feature(s) {{{', '.join(map(str, features_to_test))}}} accuracy is {accuracy:.1f}%")

                if accuracy > best_accuracy_on_level: # Updates best accuracy if current accuracy is highest seen on current level
                    best_accuracy_on_level = accuracy
                    feature_to_add_on_level = k

        if feature_to_add_on_level is not None:
            current_set_of_features.insert(0,feature_to_add_on_level) # Add feature to current set

        if best_accuracy_on_level > best_accuracy: # Compare level's accuracy to best accuracy seen out of all levels
            best_accuracy = best_accuracy_on_level
            best_set_of_features = current_set_of_features.copy()

        if i < len(data[0])-1: # Does not output during level with all features
            print(f"Feature set {{{', '.join(map(str, current_set_of_features))}}} was best, accuracy is {best_accuracy_on_level:.1f}%")
    
    print(f"Finished search!! The best feature subset is {{{', '.join(map(str, best_set_of_features))}}}, which has an accuracy of {best_accuracy:.1f}%")

def backward_elimination(data):
    # Set current set to all features and calculates accuracy with all features
    current_set_of_features = list(range(1, len(data[0])))
    best_accuracy = leave_one_out_cross_validation(data,current_set_of_features,0) # 0 because no feature to add
    print(f"Feature set {{{', '.join(map(str, current_set_of_features))}}} was best, accuracy is {best_accuracy:.1f}%")

    best_set_of_features = current_set_of_features.copy() # Inital best set starts off as all features
    for i in range(1, len(data[0])-1):
        feature_to_remove_on_level = None
        best_accuracy_on_level = 0

        for k in current_set_of_features: # Iterates through features left in current set
            features_to_test = [item for item in current_set_of_features if item != k] # Creates new list of features not including k'th feature
            accuracy = leave_one_out_cross_validation(data, features_to_test, 0)
            print(f"\tUsing feature(s) {{{', '.join(map(str, features_to_test))}}} accuracy is {accuracy:.1f}%")

            if accuracy > best_accuracy_on_level: # Updates best accuracy if current accuracy is highest seen on current level
                best_accuracy_on_level = accuracy
                feature_to_remove_on_level = k
        
        if feature_to_remove_on_level is not None:
            current_set_of_features.remove(feature_to_remove_on_level) # Remove feature from current set
        
        if best_accuracy_on_level > best_accuracy: # Compare level's accuracy to best accuracy seen out of all levels
            best_accuracy = best_accuracy_on_level
            best_set_of_features = current_set_of_features.copy()

        if i < len(data[0])-2: # Does not output on level with just one feature left
            print(f"Feature set {{{', '.join(map(str, current_set_of_features))}}} was best, accuracy is {best_accuracy_on_level:.1f}%")
    
    print(f"Finished search!! The best feature subset is {{{', '.join(map(str, best_set_of_features))}}}, which has an accuracy of {best_accuracy:.1f}%")

if __name__ == "__main__":
    main()