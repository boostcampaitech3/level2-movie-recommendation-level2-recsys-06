# Wandb Guide
wandb docs : https://docs.wandb.ai/

## Rule
중단한 실험은 모두 삭제하기.\
모델을 잘 알 수 있도록 init하기

## Install
https://wandb.ai/recsys-06/MovieLens

## wandb.init
아래와 같은 파라미터 설정 가능.\
project, entity, name, group, notes는 필수로 하자. 
```
init(
    job_type: Optional[str] = None,
    dir=None,
    config: Union[Dict, str, None] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    reinit: bool = None,
    tags: Optional[Sequence] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    magic: Union[dict, str, bool] = None,
    config_exclude_keys=None,
    config_include_keys=None,
    anonymous: Optional[str] = None,
    mode: Optional[str] = None,
    allow_val_change: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    force: Optional[bool] = None,
    tensorboard=None,
    sync_tensorboard=None,
    monitor_gym=None,
    save_code=None,
    id=None,
    settings: Union[Settings, Dict[str, Any], None] = None
) -> Union[Run, RunDisabled, None]
```

## wandb.config.update
config.update(args)를 사용하여 모든 args config에 설정. 

## wandb.log
정보들을 wandb에 저장하고 볼 수 있다.

val에서 사용되는 metric "RECALL@5", "NDCG@5", "RECALL@10", "NDCG@10"\
train에서 사용되는 metric "rec_avg_loss", "rec_cur_loss"의 기록만 하는 중\
추가 할 항목이 있으면 추가 후 PR

```
log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = None,
    sync: Optional[bool] = None
) -> None
```

## 추가내용
업뎃예정