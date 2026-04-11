import 'dart:async';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'personal_details_event.dart';
import 'personal_details_state.dart';

class PersonalDetailsBloc
    extends Bloc<PersonalDetailsEvent, PersonalDetailsState> {
  final FirebaseFirestore firestore;

  PersonalDetailsBloc({FirebaseFirestore? firestoreInstance})
      : firestore = firestoreInstance ?? FirebaseFirestore.instance,
        super(PersonalDetailsInitial()) {
    on<SubmitPersonalDetails>(_onSubmitPersonalDetails);
  }

  Future<void> _onSubmitPersonalDetails(
    SubmitPersonalDetails event,
    Emitter<PersonalDetailsState> emit,
  ) async {
    emit(PersonalDetailsSubmitting());

    try {
      // Save/update data to Firestore under collection 'personalDetails', doc with userId
      await firestore
          .collection('personalDetails')
          .doc(event.userId)
          .set(event.personalDetails.toMap(), SetOptions(merge: true));

      emit(PersonalDetailsSuccess());
    } catch (e) {
      emit(PersonalDetailsFailure(e.toString()));
    }
  }
}
