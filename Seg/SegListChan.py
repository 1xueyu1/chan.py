from Bi.BiList import CBiList
from Common.CEnum import BI_DIR, SEG_TYPE

from .EigenFX import CEigenFX
from .SegConfig import CSegConfig
from .SegListComm import CSegListComm


class CSegListChan(CSegListComm):
    def __init__(self, seg_config=CSegConfig(), lv=SEG_TYPE.BI):
        super(CSegListChan, self).__init__(seg_config=seg_config, lv=lv)

    def do_init(self):
        # 删除末尾不确定的线段
        while len(self) and not self.lst[-1].is_sure:
            _seg = self[-1]
            for bi in _seg.bi_list:
                bi.parent_seg = None
            if _seg.pre:
                _seg.pre.next = None
            self.lst.pop()
        if len(self):
            assert self.lst[-1].eigen_fx and self.lst[-1].eigen_fx.ele[-1]
            if not self.lst[-1].eigen_fx.ele[-1].lst[-1].is_sure:
                # 如果确定线段的分形的第三元素包含不确定笔，也需要重新算，不然线段分形元素的高低点可能不对
                self.lst.pop()

    def update(self, bi_lst: CBiList):
        self.do_init()
        if len(self) == 0:
            self.cal_seg_sure(bi_lst, begin_idx=0)
        else:
            self.cal_seg_sure(bi_lst, begin_idx=self[-1].end_bi.idx+1)
        self.collect_left_seg(bi_lst)

    def cal_seg_sure(self, bi_lst: CBiList, begin_idx: int):
        # 旧实现是递归推进。长周期大样本（如多年15m）会触发 RecursionError。
        # 这里改成等价的迭代推进，每次找到分型后更新起点继续算。
        cur_begin_idx = begin_idx
        max_step = max(1, len(bi_lst) * 4)
        step = 0
        while cur_begin_idx < len(bi_lst):
            step += 1
            if step > max_step:
                break

            up_eigen = CEigenFX(BI_DIR.UP, lv=self.lv)  # 上升线段下降笔
            down_eigen = CEigenFX(BI_DIR.DOWN, lv=self.lv)  # 下降线段上升笔
            last_seg_dir = None if len(self) == 0 else self[-1].dir
            next_begin_idx = None

            for bi in bi_lst[cur_begin_idx:]:
                fx_eigen = None
                if bi.is_down() and last_seg_dir != BI_DIR.UP:
                    if up_eigen.add(bi):
                        fx_eigen = up_eigen
                elif bi.is_up() and last_seg_dir != BI_DIR.DOWN:
                    if down_eigen.add(bi):
                        fx_eigen = down_eigen
                if len(self) == 0:  # 尝试确定第一段方向，不要以谁先成为分形来决定，反例：US.EVRG
                    if up_eigen.ele[1] is not None and bi.is_down():
                        last_seg_dir = BI_DIR.DOWN
                        down_eigen.clear()
                    elif down_eigen.ele[1] is not None and bi.is_up():
                        up_eigen.clear()
                        last_seg_dir = BI_DIR.UP
                    if up_eigen.ele[1] is None and last_seg_dir == BI_DIR.DOWN and bi.dir == BI_DIR.DOWN:
                        last_seg_dir = None
                    elif down_eigen.ele[1] is None and last_seg_dir == BI_DIR.UP and bi.dir == BI_DIR.UP:
                        last_seg_dir = None

                if fx_eigen:
                    next_begin_idx = self._treat_fx_eigen_iter(fx_eigen, bi_lst)
                    break

            if next_begin_idx is None:
                break
            if next_begin_idx <= cur_begin_idx:
                next_begin_idx = cur_begin_idx + 1
            cur_begin_idx = next_begin_idx

    def _treat_fx_eigen_iter(self, fx_eigen, bi_lst: CBiList):
        _test = fx_eigen.can_be_end(bi_lst)
        end_bi_idx = fx_eigen.GetPeakBiIdx()
        if _test in [True, None]:  # None表示反向分型找到尾部也没找到
            is_true = _test is not None  # 如果是正常结束
            if not self.add_new_seg(bi_lst, end_bi_idx, is_sure=is_true and fx_eigen.all_bi_is_sure()):  # 防止第一根线段的方向与首尾值异常
                return end_bi_idx + 1
            self.lst[-1].eigen_fx = fx_eigen
            if is_true:
                return end_bi_idx + 1
            return None
        else:
            return fx_eigen.lst[1].idx
