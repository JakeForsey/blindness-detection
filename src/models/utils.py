def get_first_weights(model):
    if model._get_name() =='ResNet':
        return model.conv1
    elif model._get_name() =="EfficientNet":
        return model._conv_stem
    else:
        return None
