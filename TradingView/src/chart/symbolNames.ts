// 国内期货合约代号 → 中文名映射。
// 用于图表顶部 / 搜索框 / badge 显示。
// 代号在不同交易所之间基本唯一，所以用 symbol 作 key；如果遇到冲突可改成
// `${exchange}:${symbol}` 形式。

const NAMES: Record<string, string> = {
  BTCUSDT: 'BTCUSDT 永续',
  ETHUSDT: 'ETH 永续',
  SOLUSDT: 'SOL 永续',
  SUIUSDT: 'SUI 永续',
  HYPEUSDT: 'HYPE 永续',
  ADAUSDT: 'ADA 永续',
  BNBUSDT: 'BNB 永续',
  XRPUSDT: 'XRP 永续',
  DOGEUSDT: 'DOGE 永续',
  AVAXUSDT: 'AVAX 永续',
  LINKUSDT: 'LINK 永续',

  // SHFE 上海期货交易所
  RB: '螺纹钢', HC: '热轧卷板', WR: '线材', SS: '不锈钢',
  CU: '铜', AL: '铝', ZN: '锌', PB: '铅', NI: '镍', SN: '锡',
  AU: '黄金', AG: '白银', AO: '氧化铝',
  FU: '燃油', BU: '沥青', RU: '橡胶', SP: '纸浆',

  // INE 上海能源交易中心
  SC: '原油', LU: '低硫燃料油', NR: '20号胶', BC: '国际铜', EC: '集运指数(欧线)',

  // DCE 大商所
  A: '黄大豆1号', B: '黄大豆2号', M: '豆粕', Y: '豆油',
  P: '棕榈油', C: '玉米', CS: '玉米淀粉', JD: '鸡蛋', LH: '生猪',
  L: '聚乙烯', V: 'PVC', PP: '聚丙烯',
  J: '焦炭', JM: '焦煤', I: '铁矿石',
  EG: '乙二醇', EB: '苯乙烯', RR: '粳米',
  FB: '纤维板', BB: '胶合板', PG: 'LPG液化气',

  // CZCE 郑商所
  CF: '棉花', CY: '棉纱', SR: '白糖', TA: 'PTA',
  OI: '菜籽油', MA: '甲醇', FG: '玻璃',
  SF: '硅铁', SM: '锰硅',
  WH: '强麦', RI: '早籼稻', LR: '晚籼稻', JR: '粳稻', PM: '普麦',
  RM: '菜粕', RS: '菜籽',
  SA: '纯碱', PF: '短纤', PK: '花生',
  CJ: '红枣', UR: '尿素', AP: '苹果',
  PR: '瓶片', SH: '烧碱', PX: '对二甲苯',

  // GFEX 广期所
  SI: '工业硅', LC: '碳酸锂', PS: '多晶硅',

  // CFFEX 中金所
  IF: '沪深300', IH: '上证50', IC: '中证500', IM: '中证1000',
  TS: '2年期国债', TF: '5年期国债', T: '10年期国债', TL: '30年期国债',
}

/** 给定 symbol 代号返回中文名；未知代号返回原代号本身。 */
export function symbolNameCn(symbol: string): string {
  return NAMES[symbol.toUpperCase()] ?? symbol
}
