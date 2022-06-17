""" This file includes the class and methods to read and write experiment setups. """
import dataclasses, json

@dataclasses.dataclass()
class ExperimentSetup:
    """Class for keeping track of the experiment setup."""
    name: str
    # _: dataclasses.KW_ONLY # enter data as kw atfer this
    description: str
    n_batch: int = 1
    savename: str = '' #dataclasses.field(init=False)

    def __post_init__(self):
        self.savename = f'/Users/jpetersen/rare/rfmip/experiment_setups/{self.name}.json'

    def __repr__(self):
        out_str = 'ExperimentSetup:\n'
        for key, value in self.__dict__.items():
            out_str += f'   {key}: {value}\n'
        return out_str

    def save(self):
        with open(self.savename, 'w') as fp:
            json.dump(self, fp, cls=EnhancedJSONEncoder, indent=True)


class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)


def read_exp_setup(exp_name) -> dataclasses.dataclass:
    with open(f'/Users/jpetersen/rare/rfmip/experiment_setups/{exp_name}.json') as fp:
        d = json.load(fp)
        exp = ExperimentSetup(**d)
    return exp


def main():
    exp = ExperimentSetup('test', 
    description='Ein Test',
    n_batch = 100)
    exp.save()


if __name__ == '__main__':
    main()
