import pandas as pd
import importlib
from model.dataset import parse_data,split
from Metrics import acc,pre,rec,f1
import yaml,json
from seed import set_seed

random_seed = 0
model_config='MGRD.yaml'


def enclose_class_name(value):
    if isinstance(value,dict):
        assert len(value)==1, "There can only be one class"
        for k,v in value.items():
            if k[0]==k[-1]=="_":
                return {k:v}
            else:
                return {f"_{k}_":v}
    elif isinstance(value,str):
        if value[0]==value[-1]=="_":
            return value
        else:
            return f"_{value}_"
    else:
        return value

def create_obj_from_json(js):

    if isinstance(js, dict):
        rtn_dict = {}
        for key, values in js.items():
            if need_import(key):
                assert values is None or isinstance(values,
                                                    dict), f"The value of the object {key} to be imported must be dict or None to initialize the object"
                assert len(js) == 1, f"{js} contains the {key} object to be imported and cannot contain other key-value pairs"
                key = key[1:-1]
                cls = my_import(key)
                if "__init__" in values:
                    assert isinstance(values, dict), f"__init__ keyword, put into the dictionary object, as the parent class {key} initialization function"
                    init_params = create_obj_from_json(values['__init__'])
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()
                for k, v in values.items():
                    setattr(obj, k, create_obj_from_json(v))
                return obj
            rtn_dict[key] = create_obj_from_json(values)
        return rtn_dict
    elif isinstance(js, (set, list)):
        return [create_obj_from_json(x) for x in js]
    elif isinstance(js,str):
        if need_import(js):
            cls_name = js[1:-1]
            return my_import(cls_name)()
        else:
            return js
    else:
        return js

def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls

def need_import(value):

    if isinstance(value, str) and len(value) > 3 and value[0] == value[-1] == '_' and not value == "__init__":
        return True
    else:
        return False

def update_parameters(param: dict, to_update: dict) -> dict:
    for k, v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k], (dict,)):
                    param[k].update(to_update[k])
                else:
                    param[k] = to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param


def main():

    with open(model_config, 'rb') as infile:
        cfg = yaml.safe_load(infile)

    data_dir = cfg['data_path']

    data = parse_data(data_dir)
    datalist = split(data)

    train_data = datalist[0]
    valid_data = datalist[1]
    test_data = datalist[2]

    metrics = [acc(),pre(),rec(),f1()]


    set_seed(random_seed)

    algorithm = create_obj_from_json(enclose_class_name({cfg['algorithm']:cfg['algorithm_parameters']}))


    algorithm.train(train_data, valid_data)

    pred = algorithm.predict(test_data)
    results = [m(pred, test_data) for m in metrics]
    headers = [str(m) for m in metrics]
    print(dict(zip(headers, results)))


if __name__ == '__main__':
    main()