from sklearn.tree import DecisionTreeClassifier

def bestAccuracy(X_train, X_test, y_train, y_test):
    best_acc_gini = 0
    best_acc_entropy = 0
    m_depth_gini = 0 
    min_s_leaf_gini = 0
    m_depth_entropy = 0
    min_s_leaf_entropy = 0

    # Training with criterion="gini"
    for i in range(1,20):
        for j in range(1,20):
            model = DecisionTreeClassifier(random_state=42, criterion="gini", max_depth=i, min_samples_leaf=j)
            model.fit(X_train , y_train)
            acc = model.score(X_test, y_test)

            # Recording the best scores and parameters for the best score
            if acc > best_acc_gini:
                best_acc_gini = acc
                m_depth_gini = i
                min_s_leaf_gini = j

    # Training with criterion="entropy"
    for i in range(1,20):
        for j in range(1,20):
            model = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=i, min_samples_leaf=j)
            model.fit(X_train , y_train)
            acc = model.score(X_test, y_test)

            # Recording the best scores and parameters for the best score
            if acc > best_acc_entropy:
                best_acc_entropy = acc
                m_depth_entropy = i
                min_s_leaf_entropy = j

    # Comparing "gini" with "entropy" and returning the best set of parameters
    if best_acc_entropy > best_acc_gini:
        return "entropy", m_depth_entropy,min_s_leaf_entropy
    else:
        return "gini", m_depth_gini, min_s_leaf_gini
                      