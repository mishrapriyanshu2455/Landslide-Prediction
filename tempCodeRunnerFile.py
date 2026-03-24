
print("Training model...")
mod.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred =mod.predict(X_test)


y_prob = mod.predict_proba(X_test)[:,1]
