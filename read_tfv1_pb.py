import tensorflow as tf



with tf.compat.v1.gfile.GFile("D:/Acesso_Codigos/TaskClassifier/training-Old/model/saved_model.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
        model.save('test', save_format='tf')

"""
graph_def = tf.compat.v1.GraphDef()
loaded = graph_def.ParseFromString(open("D:/Acesso_Codigos/TaskClassifier/training-Old/model/saved_model.pb",'rb').read())
print(loaded)"""