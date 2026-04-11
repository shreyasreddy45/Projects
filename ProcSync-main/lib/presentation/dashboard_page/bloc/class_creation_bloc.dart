import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../models/class_model.dart';
import 'dart:math';

part 'class_creation_event.dart';
part 'class_creation_state.dart';

class ClassCreationBloc extends Bloc<ClassCreationEvent, ClassCreationState> {
  ClassCreationBloc() : super(ClassCreationInitial()) {
    on<CreateClassEvent>(_onCreateClass);
  }

  Future<void> _onCreateClass(
      CreateClassEvent event, Emitter<ClassCreationState> emit) async {
    emit(ClassCreationLoading());

    try {
      final String classCode = _generateClassCode();

      final classModel = ClassModel(
        id: classCode, // ✅ Added this line
        code: classCode,
        name: event.name,
        section: event.section,
        createdBy: event.teacherUid,
        createdAt: DateTime.now(),
      );

      await FirebaseFirestore.instance
          .collection(
              'classrooms') // 🔁 or 'classes' if that's your actual collection
          .doc(classCode)
          .set(classModel.toMap());

      emit(ClassCreationSuccess(classModel));
    } catch (e) {
      emit(ClassCreationFailure(e.toString()));
    }
  }

  String _generateClassCode() {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    final rand = Random();
    return List.generate(6, (_) => chars[rand.nextInt(chars.length)]).join();
  }
}
