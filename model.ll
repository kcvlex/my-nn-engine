; ModuleID = 'main'
source_filename = "main"

define void @main(ptr noalias nocapture noundef readonly %0, ptr noalias nocapture noundef readonly %1) local_unnamed_addr #0 {
entry:
  %Output1 = load ptr, ptr %0, align 8
  %Input12 = load ptr, ptr %1, align 8
  %Input2 = getelementptr inbounds ptr, ptr %1, i64 1
  %Input23 = load ptr, ptr %Input2, align 8
  %Input3 = getelementptr inbounds ptr, ptr %1, i64 2
  %Input34 = load ptr, ptr %Input3, align 8
  %chunk.0 = tail call ptr @malloc(i32 480)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !0)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !3)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !5)
  %gep.ptr.1.3.1.phi.trans.insert.i = getelementptr float, ptr %Input12, i64 8
  %gep.ptr.1.1.3.phi.trans.insert.i = getelementptr float, ptr %Input12, i64 16
  %2 = load <2 x float>, ptr %gep.ptr.1.1.3.phi.trans.insert.i, align 4, !alias.scope !3, !noalias !7
  %gep.ptr.1.3.3.phi.trans.insert.i = getelementptr float, ptr %Input12, i64 18
  %3 = load <8 x float>, ptr %Input12, align 4, !alias.scope !3, !noalias !7
  %gep.ptr.0.3.1.i = getelementptr float, ptr %chunk.0, i64 8
  %4 = load <8 x float>, ptr %gep.ptr.1.3.1.phi.trans.insert.i, align 4, !alias.scope !3, !noalias !7
  %gep.ptr.0.1.3.i = getelementptr float, ptr %chunk.0, i64 16
  %5 = load <2 x float>, ptr %Input23, align 4, !alias.scope !5, !noalias !8
  %6 = shufflevector <2 x float> %5, <2 x float> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1>
  %7 = shufflevector <2 x float> %5, <2 x float> poison, <8 x i32> zeroinitializer
  %8 = fadd <8 x float> %3, %7
  store <8 x float> %8, ptr %chunk.0, align 4, !alias.scope !0, !noalias !9
  %9 = fadd <8 x float> %4, %7
  store <8 x float> %9, ptr %gep.ptr.0.3.1.i, align 4, !alias.scope !0, !noalias !9
  %10 = shufflevector <8 x float> %3, <8 x float> poison, <8 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>
  %11 = shufflevector <2 x float> %2, <2 x float> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %12 = shufflevector <8 x float> %11, <8 x float> %10, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 12, i32 13, i32 14, i32 15>
  %gep.ptr.0.4.124.i = getelementptr float, ptr %chunk.0, i64 24
  %13 = shufflevector <8 x float> %3, <8 x float> %4, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  %14 = shufflevector <2 x float> %5, <2 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %15 = fadd <8 x float> %13, %14
  store <8 x float> %15, ptr %gep.ptr.0.4.124.i, align 4, !alias.scope !0, !noalias !9
  %gep.ptr.0.2.2.1.i = getelementptr float, ptr %chunk.0, i64 32
  %16 = shufflevector <8 x float> %4, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %17 = shufflevector <2 x float> %5, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %18 = fadd <4 x float> %16, %17
  store <4 x float> %18, ptr %gep.ptr.0.2.2.1.i, align 4, !alias.scope !0, !noalias !9
  %gep.ptr.0.1.3.1.i = getelementptr float, ptr %chunk.0, i64 36
  %gep.ptr.2.2.i = getelementptr float, ptr %Input23, i64 2
  %19 = shufflevector <2 x float> %5, <2 x float> poison, <8 x i32> <i32 poison, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %20 = shufflevector <8 x float> %19, <8 x float> %3, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 8, i32 9, i32 10, i32 11>
  %gep.ptr.0.4.243.i = getelementptr float, ptr %chunk.0, i64 44
  %gep.ptr.0.2.2.2.i = getelementptr float, ptr %chunk.0, i64 52
  %gep.ptr.0.3.2.2.i = getelementptr float, ptr %chunk.0, i64 53
  %gep.ptr.0.4.2.2.i = getelementptr float, ptr %chunk.0, i64 54
  %21 = load <2 x float>, ptr %gep.ptr.2.2.i, align 4, !alias.scope !5, !noalias !8
  %22 = shufflevector <2 x float> %21, <2 x float> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1>
  %23 = shufflevector <2 x float> %21, <2 x float> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %24 = shufflevector <2 x float> %21, <2 x float> poison, <8 x i32> zeroinitializer
  %25 = fadd <8 x float> %13, %24
  store <8 x float> %25, ptr %gep.ptr.0.4.243.i, align 4, !alias.scope !0, !noalias !9
  %shift = shufflevector <8 x float> %4, <8 x float> poison, <8 x i32> <i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %26 = fadd <8 x float> %shift, %22
  %res.2.2.2.i = extractelement <8 x float> %26, i64 0
  store float %res.2.2.2.i, ptr %gep.ptr.0.2.2.2.i, align 4, !alias.scope !0, !noalias !9
  %shift128 = shufflevector <8 x float> %4, <8 x float> poison, <8 x i32> <i32 5, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %27 = fadd <8 x float> %shift128, %22
  %res.3.2.2.i = extractelement <8 x float> %27, i64 0
  store float %res.3.2.2.i, ptr %gep.ptr.0.3.2.2.i, align 4, !alias.scope !0, !noalias !9
  %28 = shufflevector <8 x float> %4, <8 x float> %3, <8 x i32> <i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 8, i32 9>
  %29 = shufflevector <8 x float> %28, <8 x float> %11, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 poison, i32 poison, i32 6, i32 7>
  %gep.ptr.0.2.i.1 = getelementptr float, ptr %chunk.0, i64 62
  %30 = shufflevector <8 x float> %3, <8 x float> %4, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  %31 = shufflevector <2 x float> %21, <2 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %32 = fadd <8 x float> %30, %31
  store <8 x float> %32, ptr %gep.ptr.0.2.i.1, align 4, !alias.scope !0, !noalias !9
  %33 = getelementptr float, ptr %chunk.0, i64 70
  %34 = shufflevector <8 x float> %4, <8 x float> poison, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison>
  %35 = shufflevector <8 x float> %34, <8 x float> %11, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 8, i32 9>
  %36 = fadd <8 x float> %35, %31
  store <8 x float> %36, ptr %33, align 4, !alias.scope !0, !noalias !9
  %gep.ptr.0.3.3.i.1 = getelementptr float, ptr %chunk.0, i64 78
  %gep.ptr.2.1.i.1 = getelementptr float, ptr %Input23, i64 4
  %37 = shufflevector <8 x float> %23, <8 x float> %3, <8 x i32> <i32 1, i32 1, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13>
  %gep.ptr.0.1.1.1.i.1 = getelementptr float, ptr %chunk.0, i64 86
  %38 = shufflevector <8 x float> %3, <8 x float> %4, <8 x i32> <i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13>
  %gep.ptr.0.4.2.1.i.1 = getelementptr float, ptr %chunk.0, i64 94
  %39 = load <2 x float>, ptr %gep.ptr.2.1.i.1, align 4, !alias.scope !5, !noalias !8
  %40 = shufflevector <2 x float> %39, <2 x float> poison, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1>
  %41 = shufflevector <2 x float> %39, <2 x float> poison, <8 x i32> zeroinitializer
  %42 = fadd <8 x float> %38, %41
  store <8 x float> %42, ptr %gep.ptr.0.1.1.1.i.1, align 4, !alias.scope !0, !noalias !9
  %gep.ptr.0.2.235.i.1 = getelementptr float, ptr %chunk.0, i64 102
  %43 = shufflevector <2 x float> %39, <2 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %44 = fadd <8 x float> %30, %43
  store <8 x float> %44, ptr %gep.ptr.0.2.235.i.1, align 4, !alias.scope !0, !noalias !9
  %45 = getelementptr float, ptr %chunk.0, i64 110
  %46 = fadd <8 x float> %35, %43
  store <8 x float> %46, ptr %45, align 4, !alias.scope !0, !noalias !9
  %gep.ptr.0.3.3.2.i.1 = getelementptr float, ptr %chunk.0, i64 118
  %47 = load <2 x float>, ptr %gep.ptr.1.3.3.phi.trans.insert.i, align 4, !alias.scope !3, !noalias !7
  %48 = shufflevector <2 x float> %47, <2 x float> poison, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %49 = shufflevector <8 x float> %12, <8 x float> %48, <8 x i32> <i32 0, i32 1, i32 8, i32 9, i32 4, i32 5, i32 6, i32 7>
  %50 = fadd <8 x float> %49, %6
  store <8 x float> %50, ptr %gep.ptr.0.1.3.i, align 4, !alias.scope !0, !noalias !9
  %51 = shufflevector <2 x float> %2, <2 x float> %47, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %52 = shufflevector <8 x float> %51, <8 x float> %23, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 poison, i32 poison, i32 poison>
  %53 = shufflevector <8 x float> %52, <8 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  %54 = fadd <8 x float> %53, %20
  store <8 x float> %54, ptr %gep.ptr.0.1.3.1.i, align 4, !alias.scope !0, !noalias !9
  %55 = shufflevector <8 x float> %29, <8 x float> %48, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 6, i32 7>
  %56 = fadd <8 x float> %55, %22
  store <8 x float> %56, ptr %gep.ptr.0.4.2.2.i, align 4, !alias.scope !0, !noalias !9
  %57 = shufflevector <2 x float> %47, <2 x float> %39, <8 x i32> <i32 0, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %58 = fadd <8 x float> %57, %37
  store <8 x float> %58, ptr %gep.ptr.0.3.3.i.1, align 4, !alias.scope !0, !noalias !9
  %59 = fadd <8 x float> %55, %40
  store <8 x float> %59, ptr %gep.ptr.0.4.2.1.i.1, align 4, !alias.scope !0, !noalias !9
  %60 = shufflevector <2 x float> %39, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  %61 = fadd <2 x float> %47, %60
  store <2 x float> %61, ptr %gep.ptr.0.3.3.2.i.1, align 4, !alias.scope !0, !noalias !9
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  %gep.ptr.2.1.3.i = getelementptr float, ptr %Input34, i64 16
  %62 = load <4 x float>, ptr %gep.ptr.2.1.3.i, align 4, !alias.scope !15, !noalias !17
  %63 = shufflevector <4 x float> %62, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
  %64 = load <8 x float>, ptr %Input34, align 4, !alias.scope !15, !noalias !17
  %65 = shufflevector <8 x float> %64, <8 x float> poison, <8 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>
  %66 = shufflevector <8 x float> %63, <8 x float> %65, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
  %gep.ptr.2.3.1.phi.trans.insert.i = getelementptr float, ptr %Input34, i64 8
  %67 = load <8 x float>, ptr %gep.ptr.2.3.1.phi.trans.insert.i, align 4, !alias.scope !15, !noalias !17
  %68 = shufflevector <8 x float> %67, <8 x float> poison, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison>
  %69 = shufflevector <8 x float> %68, <8 x float> %63, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %70 = shufflevector <8 x float> %64, <8 x float> %67, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  %71 = load <8 x float>, ptr %chunk.0, align 4, !alias.scope !13, !noalias !18
  %72 = fadd <8 x float> %64, %71
  store <8 x float> %72, ptr %Output1, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.3.1.i = getelementptr float, ptr %chunk.0, i64 8
  %gep.ptr.0.3.1.i37 = getelementptr float, ptr %Output1, i64 8
  %73 = load <8 x float>, ptr %gep.ptr.1.3.1.i, align 4, !alias.scope !13, !noalias !18
  %74 = fadd <8 x float> %67, %73
  store <8 x float> %74, ptr %gep.ptr.0.3.1.i37, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.1.3.i = getelementptr float, ptr %chunk.0, i64 16
  %gep.ptr.0.1.3.i55 = getelementptr float, ptr %Output1, i64 16
  %75 = load <8 x float>, ptr %gep.ptr.1.1.3.i, align 4, !alias.scope !13, !noalias !18
  %76 = fadd <8 x float> %66, %75
  store <8 x float> %76, ptr %gep.ptr.0.1.3.i55, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.4.131.i = getelementptr float, ptr %chunk.0, i64 24
  %gep.ptr.0.4.136.i = getelementptr float, ptr %Output1, i64 24
  %77 = load <8 x float>, ptr %gep.ptr.1.4.131.i, align 4, !alias.scope !13, !noalias !18
  %78 = fadd <8 x float> %70, %77
  store <8 x float> %78, ptr %gep.ptr.0.4.136.i, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.2.2.1.i = getelementptr float, ptr %chunk.0, i64 32
  %gep.ptr.0.2.2.1.i84 = getelementptr float, ptr %Output1, i64 32
  %79 = load <8 x float>, ptr %gep.ptr.1.2.2.1.i, align 4, !alias.scope !13, !noalias !18
  %80 = fadd <8 x float> %69, %79
  store <8 x float> %80, ptr %gep.ptr.0.2.2.1.i84, align 4, !alias.scope !10, !noalias !19
  %81 = getelementptr float, ptr %chunk.0, i64 40
  %82 = getelementptr float, ptr %Output1, i64 40
  %83 = load <8 x float>, ptr %81, align 4, !alias.scope !13, !noalias !18
  %84 = fadd <8 x float> %64, %83
  store <8 x float> %84, ptr %82, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.3.1.2.i = getelementptr float, ptr %chunk.0, i64 48
  %gep.ptr.0.3.1.2.i107 = getelementptr float, ptr %Output1, i64 48
  %85 = load <8 x float>, ptr %gep.ptr.1.3.1.2.i, align 4, !alias.scope !13, !noalias !18
  %86 = fadd <8 x float> %67, %85
  store <8 x float> %86, ptr %gep.ptr.0.3.1.2.i107, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.1.3.2.i = getelementptr float, ptr %chunk.0, i64 56
  %gep.ptr.0.1.3.2.i119 = getelementptr float, ptr %Output1, i64 56
  %87 = load <4 x float>, ptr %gep.ptr.1.1.3.2.i, align 4, !alias.scope !13, !noalias !18
  %88 = fadd <4 x float> %62, %87
  store <4 x float> %88, ptr %gep.ptr.0.1.3.2.i119, align 4, !alias.scope !10, !noalias !19
  %89 = getelementptr float, ptr %chunk.0, i64 60
  %90 = getelementptr float, ptr %Output1, i64 60
  %91 = load <8 x float>, ptr %89, align 4, !alias.scope !13, !noalias !18
  %92 = fadd <8 x float> %64, %91
  store <8 x float> %92, ptr %90, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.3.1.i.1 = getelementptr float, ptr %chunk.0, i64 68
  %gep.ptr.0.3.1.i37.1 = getelementptr float, ptr %Output1, i64 68
  %93 = load <8 x float>, ptr %gep.ptr.1.3.1.i.1, align 4, !alias.scope !13, !noalias !18
  %94 = fadd <8 x float> %67, %93
  store <8 x float> %94, ptr %gep.ptr.0.3.1.i37.1, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.1.3.i.1 = getelementptr float, ptr %chunk.0, i64 76
  %gep.ptr.0.1.3.i55.1 = getelementptr float, ptr %Output1, i64 76
  %95 = load <8 x float>, ptr %gep.ptr.1.1.3.i.1, align 4, !alias.scope !13, !noalias !18
  %96 = fadd <8 x float> %66, %95
  store <8 x float> %96, ptr %gep.ptr.0.1.3.i55.1, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.4.131.i.1 = getelementptr float, ptr %chunk.0, i64 84
  %gep.ptr.0.4.136.i.1 = getelementptr float, ptr %Output1, i64 84
  %97 = load <8 x float>, ptr %gep.ptr.1.4.131.i.1, align 4, !alias.scope !13, !noalias !18
  %98 = fadd <8 x float> %70, %97
  store <8 x float> %98, ptr %gep.ptr.0.4.136.i.1, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.2.2.1.i.1 = getelementptr float, ptr %chunk.0, i64 92
  %gep.ptr.0.2.2.1.i84.1 = getelementptr float, ptr %Output1, i64 92
  %99 = load <8 x float>, ptr %gep.ptr.1.2.2.1.i.1, align 4, !alias.scope !13, !noalias !18
  %100 = fadd <8 x float> %69, %99
  store <8 x float> %100, ptr %gep.ptr.0.2.2.1.i84.1, align 4, !alias.scope !10, !noalias !19
  %101 = getelementptr float, ptr %chunk.0, i64 100
  %102 = getelementptr float, ptr %Output1, i64 100
  %103 = load <8 x float>, ptr %101, align 4, !alias.scope !13, !noalias !18
  %104 = fadd <8 x float> %64, %103
  store <8 x float> %104, ptr %102, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.3.1.2.i.1 = getelementptr float, ptr %chunk.0, i64 108
  %gep.ptr.0.3.1.2.i107.1 = getelementptr float, ptr %Output1, i64 108
  %105 = load <8 x float>, ptr %gep.ptr.1.3.1.2.i.1, align 4, !alias.scope !13, !noalias !18
  %106 = fadd <8 x float> %67, %105
  store <8 x float> %106, ptr %gep.ptr.0.3.1.2.i107.1, align 4, !alias.scope !10, !noalias !19
  %gep.ptr.1.1.3.2.i.1 = getelementptr float, ptr %chunk.0, i64 116
  %gep.ptr.0.1.3.2.i119.1 = getelementptr float, ptr %Output1, i64 116
  %107 = load <4 x float>, ptr %gep.ptr.1.1.3.2.i.1, align 4, !alias.scope !13, !noalias !18
  %108 = fadd <4 x float> %62, %107
  store <4 x float> %108, ptr %gep.ptr.0.1.3.2.i119.1, align 4, !alias.scope !10, !noalias !19
  tail call void @free(ptr nonnull %chunk.0)
  ret void
}

declare noalias ptr @malloc(i32) local_unnamed_addr

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr nocapture noundef) local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #2

attributes #0 = { "target-cpu"="alderlake" "target-features"="+prfchw,-cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,+xsaves,-avx512fp16,-usermsr,-sm4,+sse4.1,-avx512ifma,+xsave,-avx512pf,+sse4.2,-tsxldtrk,+ptwrite,+widekl,-sm3,+invpcid,+64bit,+xsavec,-avx10.1-512,-avx512vpopcntdq,+cmov,-avx512vp2intersect,-avx512cd,+movbe,-avxvnniint8,-avx512er,-amx-int8,+kl,-avx10.1-256,-sha512,+avxvnni,-rtm,+adx,+avx2,+hreset,+movdiri,+serialize,+vpclmulqdq,-avx512vl,-uintr,+clflushopt,-raoint,-cmpccxadd,+bmi,-amx-tile,+sse,+gfni,-avxvnniint16,-amx-fp16,+xsaveopt,+rdrnd,-avx512f,-amx-bf16,-avx512bf16,-avx512vnni,+cx8,-avx512bw,+sse3,+pku,+fsgsbase,-clzero,-mwaitx,-lwp,+lzcnt,+sha,+movdir64b,-wbnoinvd,-enqcmd,-prefetchwt1,-avxneconvert,-tbm,+pconfig,-amx-complex,+ssse3,+cx16,+bmi2,+fma,+popcnt,-avxifma,+f16c,-avx512bitalg,-rdpru,+clwb,+mmx,+sse2,+rdseed,-avx512vbmi2,-prefetchi,+rdpid,-fma4,-avx512vbmi,+shstk,+vaes,+waitpkg,-sgx,+fxsr,-avx512dq,-sse4a" }
attributes #1 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!0 = !{!1}
!1 = distinct !{!1, !2, !"Add1: argument 0"}
!2 = distinct !{!2, !"Add1"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"Add1: argument 1"}
!5 = !{!6}
!6 = distinct !{!6, !2, !"Add1: argument 2"}
!7 = !{!1, !6}
!8 = !{!1, !4}
!9 = !{!4, !6}
!10 = !{!11}
!11 = distinct !{!11, !12, !"Add2: argument 0"}
!12 = distinct !{!12, !"Add2"}
!13 = !{!14}
!14 = distinct !{!14, !12, !"Add2: argument 1"}
!15 = !{!16}
!16 = distinct !{!16, !12, !"Add2: argument 2"}
!17 = !{!11, !14}
!18 = !{!11, !16}
!19 = !{!14, !16}
