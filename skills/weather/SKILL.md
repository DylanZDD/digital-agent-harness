---
name: weather
description: Get current weather and forecasts (no API key required).
---

# Weather
Two free services, no API keys needed.

# Guidelines
1.先确认是否有明确的地点
2.确认有明确地点之后，根据地点进行天气查询，根据问题选择不同的查询方式。
  a.如果用户想要知道今天的天气，那就是用 `wttr.in 格式=4`。
  b.如果用户想知道未来几天的天气，则使用 `wttr.in ?T`。
  c.如果用户想要知道今天的天气变化，以小时为颗粒度的。那就使用 `wttr.in 格式=v2`
3.根据查询结果，比较全面的展示天气情况，包括温度、天气情况、风速风向等。
4.根据具体的天气情况，表达对用户的关系。例如下雨天气提醒用户拿伞，温度低于10度，提心用户多穿衣服等

# wttr.in (primary)
## 查询天气
```bash
curl -s "wttr.in/Shanghai?format=4"
```

## 查询当天天气（细粒度）
```bash
curl -s "wttr.in/Shanghai?format=v2"
```

## 查询未来几天天气
```bash
curl -s "wttr.in/Shanghai?T
```